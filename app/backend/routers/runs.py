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
AIRFLOW_SERVICE = os.getenv("AIRFLOW_SERVICE", "my-airflow-api-server")
AIRFLOW_NAMESPACE = os.getenv("AIRFLOW_NAMESPACE", "airflow")
AIRFLOW_PORT = os.getenv("AIRFLOW_PORT", "8080")
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
    processed_table: str              # e.g. "iceberg.processed.network_traffic_raw_20260101_120000"
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
    def validate_splits_no_overlap(self) -> "RunRequest":
        """Validate split dates: format, positive duration, and no pairwise overlap."""

        def _parse(s: str, label: str) -> datetime:
            try:
                return datetime.strptime(s, _DATE_FMT)
            except ValueError:
                raise ValueError(
                    f"{label} must be YYYY-MM-DD HH:MM:SS, got '{s}'"
                )

        parsed: dict[str, tuple[datetime, datetime]] = {
            "train": (
                _parse(self.splits.train.start, "splits.train.start"),
                _parse(self.splits.train.end,   "splits.train.end"),
            ),
            "val": (
                _parse(self.splits.val.start, "splits.val.start"),
                _parse(self.splits.val.end,   "splits.val.end"),
            ),
            "test": (
                _parse(self.splits.test.start, "splits.test.start"),
                _parse(self.splits.test.end,   "splits.test.end"),
            ),
        }

        # Each split must have positive duration
        for name, (s, e) in parsed.items():
            if s >= e:
                raise ValueError(
                    f"splits.{name}.start must be before splits.{name}.end"
                )

        # No overlap between any two splits
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        overlapping = []
        for a, b in pairs:
            a_start, a_end = parsed[a]
            b_start, b_end = parsed[b]
            if a_start < b_end and b_start < a_end:
                overlapping.append(f"{a} and {b}")
        if overlapping:
            raise ValueError(
                f"Splits overlap — conflicting pairs: {', '.join(overlapping)}"
            )

        return self


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _airflow_request(method: str, path: str, body=None):
    """Make an Airflow API call via the k8s client.

    Airflow 3.x requires JWT Bearer auth. We first POST to /auth/token with
    username/password in the request body, then use the returned access_token
    as a Bearer header for the actual API call.

    - In-cluster: direct HTTP using cluster-local DNS.
    - Local (kubeconfig): route through the k8s API-server proxy so the
      cluster-internal service name resolves (k3s uses client-cert auth).
    """
    import json as _json
    from kubernetes import client as _k8s_client, config as _k8s_config

    in_cluster = False
    try:
        _k8s_config.load_incluster_config()
        in_cluster = True
    except Exception:
        _k8s_config.load_kube_config()

    if in_cluster:
        base_url = (
            f"http://{AIRFLOW_SERVICE}.{AIRFLOW_NAMESPACE}.svc.cluster.local"
            f":{AIRFLOW_PORT}"
        )
        # Airflow 3.x: obtain JWT token
        token_resp = _requests.post(
            f"{base_url}/auth/token",
            json={"username": AIRFLOW_USER, "password": AIRFLOW_PASSWORD},
            timeout=15,
        )
        token = token_resp.json().get("access_token", "")

        resp = _requests.request(
            method.upper(),
            f"{base_url}/{path.lstrip('/')}",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
            timeout=15,
        )
        try:
            data = resp.json()
        except ValueError:
            data = {"raw": resp.text}
        return resp.status_code, data

    # Local dev: proxy via k8s API server (k3s uses client-cert auth)
    api = _k8s_client.ApiClient()

    def _proxy(m: str, p: str, b=None, extra_headers: dict | None = None):
        proxy_url = (
            f"{api.configuration.host.rstrip('/')}"
            f"/api/v1/namespaces/{AIRFLOW_NAMESPACE}/services"
            f"/{AIRFLOW_SERVICE}:{AIRFLOW_PORT}/proxy/{p.lstrip('/')}"
        )
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        return api.rest_client.pool_manager.request(
            m.upper(),
            proxy_url,
            headers=headers,
            body=_json.dumps(b).encode("utf-8") if b is not None else None,
            timeout=15.0,
        )

    # Airflow 3.x: obtain JWT token through the proxy
    token_resp = _proxy("POST", "auth/token", b={"username": AIRFLOW_USER, "password": AIRFLOW_PASSWORD})
    token = _json.loads(token_resp.data).get("access_token", "")

    http_resp = _proxy(method, path, b=body, extra_headers={"Authorization": f"Bearer {token}"})
    try:
        data = _json.loads(http_resp.data)
    except (ValueError, AttributeError):
        data = {"raw": (http_resp.data or b"").decode("utf-8", errors="replace")}
    return http_resp.status, data


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


def _generate_training_params_yaml(req: RunRequest, execution_id: str, cfg: dict) -> str:
    """Build params_training.yaml: solo config de entrenamiento (sin spark, sin serving/canary)."""
    mlflow_cfg = cfg.get("mlflow", {})

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
            "tuning": {
                "enabled": req.tuning.enabled,
                "number_of_trials": req.tuning.number_of_trials,
            },
            "skip_preprocessing": True,
            "processed_table": req.processed_table,
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
        # kuberay.model — config de entrenamiento y registro MLflow
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
        },
        "iceberg_tables": {
            "warehouse": f"s3://{S3_BUCKET}/warehouse",
            "metadata": {
                "catalog": "iceberg",
                "namespace": "metadata",
                "table": "preprocessing_artifacts",
                "full_name": "iceberg.metadata.preprocessing_artifacts",
            },
            "processed": {
                "catalog": "iceberg",
                "namespace": "processed",
            },
        },
    }

    return _yaml.dump(params, default_flow_style=False, allow_unicode=True)


def _generate_serving_params_yaml(execution_id: str, cfg: dict) -> str:
    """Build params_serving.yaml: config de despliegue Ray Serve (alias, canary, webhook)."""
    serving_cfg = cfg.get("serving", {})

    params: dict[str, Any] = {
        "execution": {
            "execution_id": execution_id,
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
    training_yaml_str = _generate_training_params_yaml(request, execution_id, cfg)
    serving_yaml_str = _generate_serving_params_yaml(execution_id, cfg)

    # Upload params_training.yaml y params_serving.yaml a S3
    s3 = _s3_client()

    training_key = f"params/{execution_id}/params_training.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=training_key,
        Body=training_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    train_params_s3_path = f"s3://{S3_BUCKET}/{training_key}"

    serving_key = f"params/{execution_id}/params_serving.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=serving_key,
        Body=serving_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    serving_params_s3_path = f"s3://{S3_BUCKET}/{serving_key}"

    # Trigger Airflow DAG (training_only — Spark preprocessing skipped)
    dag_conf: dict[str, Any] = {
        "execution_id": execution_id,
        "raw_table": "",
        "dsl_s3_path": "",
        "train_params_s3_path": train_params_s3_path,
        "serving_params_s3_path": serving_params_s3_path,
        "model_type": request.framework,
        "pipeline_mode": "training_only",
        "processed_table": request.processed_table,
    }
    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{DAG_ID}/dagRuns",
            body={"logical_date": datetime.now(timezone.utc).isoformat(), "conf": dag_conf},
        )
        if status >= 400:
            raise HTTPException(status_code=502, detail=f"Airflow returned {status}: {data}")
        dag_run_id = data.get("dag_run_id", "")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger Airflow DAG: {exc}",
        ) from exc

    return {
        "dag_run_id": dag_run_id,
        "execution_id": execution_id,
        "train_params_s3_path": train_params_s3_path,
        "serving_params_s3_path": serving_params_s3_path,
    }


@router.get("/{dag_run_id}/status")
async def run_status(dag_run_id: str):
    """Poll the state of an Airflow DAG run."""
    try:
        status, data = _airflow_request(
            "GET", f"api/v2/dags/{DAG_ID}/dagRuns/{dag_run_id}"
        )
        if status >= 400:
            raise HTTPException(status_code=502, detail=f"Airflow returned {status}: {data}")
    except HTTPException:
        raise
    except Exception as exc:
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
