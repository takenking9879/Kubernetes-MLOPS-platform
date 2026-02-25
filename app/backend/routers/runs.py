"""
Run pipeline router.

Endpoints:
  GET  /api/v2/runs/check                  - check if artifact_set_id (execution_id) already exists
  POST /api/v2/runs                        - generate params.yaml, upload to S3, trigger Airflow DAG
  GET  /api/v2/runs/{dag_run_id}/status    - poll Airflow DAG run state
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import boto3
import requests as _requests
import yaml as _yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/api/v2/runs", tags=["runs"])

# ─── Config ──────────────────────────────────────────────────────────────────

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AIRFLOW_BASE_URL = os.getenv(
    "AIRFLOW_BASE_URL",
    "http://airflow.airflow.svc.cluster.local:8080",
)
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "ml_pipeline"

# Path to k3s/config.yaml (static infra config — used for backward-compat block)
_REPO_CONFIG_PATH = os.getenv(
    "REPO_CONFIG_PATH",
    "/git-data/repo/k3s/config.yaml",
)

# ─── Hyperparameter validation data ──────────────────────────────────────────
# Mirrors src/schemas/model/*_params.py exactly.
# Defined here to avoid cross-package import issues at the boundary layer.

_XGBOOST_ALLOWED_KEYS: frozenset = frozenset({
    "num_boost_round", "objective", "eval_metric", "booster",
    "tree_method", "verbosity", "eta", "max_depth", "min_child_weight",
    "subsample", "lambda", "alpha",
})

_PYTORCH_ALLOWED_KEYS: frozenset = frozenset({
    "batch_size", "max_epochs", "lr", "weight_decay",
})

# Expected Python types for each parameter (int/float overlap handled explicitly)
_XGBOOST_PARAM_TYPES: dict[str, type] = {
    "num_boost_round": int, "objective": str, "eval_metric": list,
    "booster": str, "tree_method": str, "verbosity": int,
    "eta": float, "max_depth": int, "min_child_weight": int,
    "subsample": float, "lambda": float, "alpha": float,
}
_PYTORCH_PARAM_TYPES: dict[str, type] = {
    "batch_size": int, "max_epochs": int, "lr": float, "weight_decay": float,
}

_XGBOOST_TUNE_SETTINGS_DEFAULTS: dict[str, int] = {
    "num_boost_round": 10,
    "grace_period": 5,
    "reduction_factor": 2,
}
_PYTORCH_TUNE_SETTINGS_DEFAULTS: dict[str, int] = {
    "grace_period": 5,
    "reduction_factor": 2,
    "max_epochs": 10,
}

_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


def _load_static_config() -> dict[str, Any]:
    """Load k3s/config.yaml for use in the backward-compat block of params.yaml."""
    path = Path(_REPO_CONFIG_PATH)
    if path.exists():
        with open(path) as fh:
            return _yaml.safe_load(fh) or {}
    return {}


# ─── Models ──────────────────────────────────────────────────────────────────


class SplitRange(BaseModel):
    start: str
    end: str


class Splits(BaseModel):
    train: SplitRange
    val: SplitRange
    test: SplitRange


class TuningConfig(BaseModel):
    enabled: bool = True
    number_of_trials: int = Field(3, ge=1)


class ModelConfig(BaseModel):
    experiment_name: str = "kuberay-attack-detection"
    registry_model_name: str = "attack-detection"
    target: str = "attack"
    num_classes: int = Field(6, ge=2)
    seed: int = 42


class RunRequest(BaseModel):
    dataset: str
    dsl_version: int
    execution_id: str = ""
    framework: Literal["xgboost", "pytorch"] = "xgboost"
    splits: Splits
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sample_fraction_for_tuning: float = Field(0.2, ge=0.01, le=1.0)
    hyperparams: dict[str, Any] = {}       # *_PARAMS overrides
    tune_settings: dict[str, Any] = {}    # *_TUNE_SETTINGS overrides

    @model_validator(mode="after")
    def validate_hyperparams(self) -> "RunRequest":
        """Validate hyperparameter keys and types against the allowed set."""
        fw = self.framework
        allowed = _XGBOOST_ALLOWED_KEYS if fw == "xgboost" else _PYTORCH_ALLOWED_KEYS
        expected_types = _XGBOOST_PARAM_TYPES if fw == "xgboost" else _PYTORCH_PARAM_TYPES

        for key, val in self.hyperparams.items():
            if key not in allowed:
                raise ValueError(
                    f"hyperparams['{key}'] is not a valid {fw} parameter. "
                    f"Allowed keys: {sorted(allowed)}"
                )
            exp_type = expected_types.get(key)
            if exp_type is not None and not isinstance(val, exp_type):
                # Allow int where float is expected (JSON numbers)
                if exp_type is float and isinstance(val, int):
                    pass
                else:
                    raise ValueError(
                        f"hyperparams['{key}'] expected {exp_type.__name__}, "
                        f"got {type(val).__name__}"
                    )

        return self

    @model_validator(mode="after")
    def validate_splits_format_and_order(self) -> "RunRequest":
        """Validate split dates: format + chronological order (train < val < test)."""

        def _parse(s: str, label: str) -> datetime:
            try:
                return datetime.strptime(s, _DATE_FMT)
            except ValueError:
                raise ValueError(
                    f"{label} must be YYYY-MM-DD HH:MM:SS, got '{s}'"
                )

        train_start = _parse(self.splits.train.start, "splits.train.start")
        train_end   = _parse(self.splits.train.end,   "splits.train.end")
        val_start   = _parse(self.splits.val.start,   "splits.val.start")
        val_end     = _parse(self.splits.val.end,     "splits.val.end")
        test_start  = _parse(self.splits.test.start,  "splits.test.start")
        test_end    = _parse(self.splits.test.end,    "splits.test.end")

        if train_start >= train_end:
            raise ValueError("splits.train.start must be before splits.train.end")
        if train_end > val_start:
            raise ValueError("splits.train.end must not exceed splits.val.start")
        if val_start >= val_end:
            raise ValueError("splits.val.start must be before splits.val.end")
        if val_end > test_start:
            raise ValueError("splits.val.end must not exceed splits.test.start")
        if test_start >= test_end:
            raise ValueError("splits.test.start must be before splits.test.end")

        return self


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_dsl_s3_path(dataset: str, version: int) -> str:
    """Find the S3 key for dsl/dsl_{dataset}/v{version}__*.yaml."""
    s3 = _s3_client()
    prefix = f"dsl/dsl_{dataset}/v{version}__"
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=5)
    objs = [o for o in resp.get("Contents", []) if o["Key"].endswith(".yaml")]
    if not objs:
        raise HTTPException(
            status_code=404,
            detail=f"DSL version {version} not found for dataset '{dataset}' in S3.",
        )
    return f"s3://{S3_BUCKET}/{objs[0]['Key']}"


def _generate_params_yaml(req: RunRequest, execution_id: str, cfg: dict) -> str:
    """Build the execution params.yaml dict and return as a YAML string."""
    mlflow_cfg = cfg.get("mlflow", {})
    serving_cfg = cfg.get("serving", {})

    dsl_s3_path = _resolve_dsl_s3_path(req.dataset, req.dsl_version)

    # ── Resolved *_TUNE_SETTINGS (merge defaults + overrides) ─────────────
    tune_defaults = (
        _XGBOOST_TUNE_SETTINGS_DEFAULTS if req.framework == "xgboost"
        else _PYTORCH_TUNE_SETTINGS_DEFAULTS
    )
    resolved_tune_settings = {**tune_defaults, **req.tune_settings}

    # ── Hyperparams block ──────────────────────────────────────────────────
    hyperparams_block: dict[str, Any] = {
        req.framework: req.hyperparams,
    }
    if req.tuning.enabled:
        hyperparams_block["tuning"] = resolved_tune_settings

    params: dict[str, Any] = {
        "execution": {
            "execution_id": execution_id,
            "raw_table": f"iceberg.raw.{req.dataset}",
            "dsl_s3_path": dsl_s3_path,
            "tuning": {
                "enabled": req.tuning.enabled,
                "number_of_trials": req.tuning.number_of_trials,
            },
        },
        "splits": {
            "train": {"start": req.splits.train.start, "end": req.splits.train.end},
            "val":   {"start": req.splits.val.start,   "end": req.splits.val.end},
            "test":  {"start": req.splits.test.start,  "end": req.splits.test.end},
        },
        "model": {
            "framework": req.framework,
            "experiment_name": req.model.experiment_name,
            "registry_model_name": req.model.registry_model_name,
            "target": req.model.target,
            "num_classes": req.model.num_classes,
            "tune": req.tuning.enabled,
            "sample_fraction_for_tuning": req.sample_fraction_for_tuning,
            "seed": req.model.seed,
        },
        "hyperparams": hyperparams_block,
        # ── Backward-compat block (serving layer reads kuberay.*) ──────────
        "kuberay": {
            "model": {
                "tune": req.tuning.enabled,
                "sample_fraction_for_tuning": req.sample_fraction_for_tuning,
                "target": req.model.target,
                "num_classes": req.model.num_classes,
                "dsl_count_dim": True,
                "input_dim": 14,
                "framework": req.framework,
                "seed": req.model.seed,
                "mlflow_tracking_uri": mlflow_cfg.get("tracking_uri", "http://my-mlflow"),
                "mlflow_experiment_name": req.model.experiment_name,
                "mlflow_artifact_location": mlflow_cfg.get(
                    "artifact_base", f"s3://{S3_BUCKET}/mlflow-artifacts/"
                ),
                "mlflow_registry_model_name": req.model.registry_model_name,
            },
            "serving": {
                "alias": serving_cfg.get("alias", "champion"),
                "canary": False,
                "webhook_public_base_url": serving_cfg.get("webhook_base_url", ""),
                "webhook_path": serving_cfg.get("webhook_path", "/infer/webhook"),
                "webhook_name": serving_cfg.get("webhook_name", ""),
                "webhook_max_timestamp_age_seconds": serving_cfg.get(
                    "webhook_max_timestamp_age_seconds", 300
                ),
            },
            "canary": {
                "alias": serving_cfg.get("canary_alias", "challenger"),
                "canary_probability": serving_cfg.get("canary_probability", 0.10),
                "initial_replicas": 0,
            },
        },
        "spark": {
            "app_name": "spark-preprocessing",
            "bucket": S3_BUCKET,
            "read_batch_size": 512,
            "write_batch_size": 100000,
            "num_classes": req.model.num_classes,
            "target": req.model.target,
            "schemas": {
                "input_schema": "kafka_schema_features",
                "features_schema": "schema_features",
                "output_schema": "prediction_schema",
                "preprocessed_schema": "schema_preprocessed",
                "full_schema": "schema_full",
            },
            "converters": {
                "kafka_to_features": "kafka_to_schema_features",
            },
            "prediction": {
                "type": "ray_serve",
                "ray_serve": {
                    "url": (serving_cfg.get("webhook_base_url", "http://model-serving-serve-svc.ray.svc.cluster.local:8000") + "/infer"),
                    "batch_size": 256,
                    "timeout": 30,
                    "max_retries": 3,
                    "request_payload_format": "list",
                    "response_format": "json",
                    "prediction_key": "predictions",
                },
                "columns": {
                    "id_column": "event_id",
                    "prediction_column": "label",
                },
            },
            "output": {
                "format": "json",
                "key_column": "event_id",
                "value_columns": ["timestamp", "event_id", "properties", "label"],
            },
            "checkpoint": {
                "location": f"s3a://{S3_BUCKET}/checkpoints/kafka-spark-inference",
                "cleanup_on_exit": False,
            },
            "operational": {
                "check_kafka_connection": True,
                "log_level": "INFO",
            },
            "streaming": {
                "online_processing_time": "250 milliseconds",
                "starting_offsets": "latest",
                "fail_on_data_loss": False,
            },
        },
        "iceberg_tables": {
            "warehouse": f"s3://{S3_BUCKET}/warehouse",
            "raw": {
                "catalog": "iceberg",
                "namespace": "raw",
                "table": req.dataset,
                "full_name": f"iceberg.raw.{req.dataset}",
            },
            "metadata": {
                "catalog": "iceberg",
                "namespace": "metadata",
                "table": "preprocessing_artifacts",
                "full_name": "iceberg.metadata.preprocessing_artifacts",
            },
            "processed": {
                "catalog": "iceberg",
                "namespace": "processed",
                # Tables created as: iceberg.processed.{dataset}_{execution_id}
            },
        },
    }

    return _yaml.dump(params, default_flow_style=False, allow_unicode=True)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/check")
async def check_artifact(execution_id: str, dataset: str):
    """Check if an artifact_set_id already exists in the Iceberg metadata table."""
    try:
        from pyiceberg.catalog.glue import GlueCatalog

        catalog = GlueCatalog(
            "glue",
            **{
                "s3.region": AWS_REGION,
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            },
        )
        table = catalog.load_table("metadata.preprocessing_artifacts")
        df = table.scan(
            row_filter=f"artifact_set_id = '{execution_id}'",
            limit=1,
            selected_fields=("artifact_set_id", "processed_table_name"),
        ).to_pandas()
        exists = len(df) > 0
        processed_table = df.iloc[0]["processed_table_name"] if exists else None
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query Iceberg metadata table: {exc}",
        ) from exc

    return {"exists": exists, "processed_table": processed_table}


@router.post("", status_code=201)
async def submit_run(request: RunRequest):
    """Generate params.yaml, upload to S3, and trigger the Airflow ml_pipeline DAG."""

    execution_id = request.execution_id.strip() or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    cfg = _load_static_config()
    params_yaml_str = _generate_params_yaml(request, execution_id, cfg)

    # Upload params.yaml to S3
    s3 = _s3_client()
    params_key = f"params/{execution_id}/params.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=params_key,
        Body=params_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    params_s3_path = f"s3://{S3_BUCKET}/{params_key}"

    dsl_s3_path = _resolve_dsl_s3_path(request.dataset, request.dsl_version)

    # Trigger Airflow DAG
    airflow_url = f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v1/dags/{DAG_ID}/dagRuns"
    try:
        resp = _requests.post(
            airflow_url,
            json={
                "conf": {
                    "execution_id": execution_id,
                    "raw_table": f"iceberg.raw.{request.dataset}",
                    "dsl_s3_path": dsl_s3_path,
                    "params_s3_path": params_s3_path,
                    "model_type": request.framework,
                },
            },
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            timeout=15,
        )
        resp.raise_for_status()
        dag_run_id = resp.json().get("dag_run_id", "")
    except _requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger Airflow DAG: {exc}",
        ) from exc

    return {
        "dag_run_id": dag_run_id,
        "execution_id": execution_id,
        "params_s3_path": params_s3_path,
        "dsl_s3_path": dsl_s3_path,
    }


@router.get("/{dag_run_id}/status")
async def run_status(dag_run_id: str):
    """Poll the state of an Airflow DAG run."""
    airflow_url = (
        f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v1/dags/{DAG_ID}/dagRuns/{dag_run_id}"
    )
    try:
        resp = _requests.get(
            airflow_url,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except _requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch Airflow DAG run status: {exc}",
        ) from exc

    return {
        "dag_run_id": dag_run_id,
        "state": data.get("state", "unknown"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
    }
