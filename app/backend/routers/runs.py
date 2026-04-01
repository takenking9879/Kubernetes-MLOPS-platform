"""
Run pipeline router.

Endpoints:
  GET  /api/v2/runs/check                  - check if artifact_set_id (execution_id) already exists
  POST /api/v2/runs                        - generate params.yaml, upload to S3, trigger Airflow DAG
  GET  /api/v2/runs/{dag_run_id}/status    - poll Airflow DAG run state
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import boto3
import requests as _requests
import yaml as _yaml
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from services.resource_constraints import (
    build_any_of_from_constraints,
    has_explicit_resource_selection,
)

router = APIRouter(prefix="/api/v2/runs", tags=["runs"])

# ─── Config ──────────────────────────────────────────────────────────────────

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AIRFLOW_SERVICE = os.getenv("AIRFLOW_SERVICE", "my-airflow-api-server")
AIRFLOW_NAMESPACE = os.getenv("AIRFLOW_NAMESPACE", "airflow")
AIRFLOW_PORT = os.getenv("AIRFLOW_PORT", "8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "training_pipeline"
DAG_ID_SKYPILOT = "training_pipeline_skypilot"

# Path to k3s/config.yaml (static infra config)
_REPO_CONFIG_PATH = os.getenv(
    "REPO_CONFIG_PATH",
    "/git-data/repo/k3s/config.yaml",
)

# ─── Hyperparameter validation data ──────────────────────────────────────────

_XGBOOST_ALLOWED_KEYS: frozenset = frozenset({
    "num_boost_round", "objective", "eval_metric", "booster",
    "tree_method", "verbosity", "eta", "max_depth", "min_child_weight",
    "subsample", "lambda", "alpha",
})

_PYTORCH_ALLOWED_KEYS: frozenset = frozenset({
    "batch_size", "max_epochs", "lr", "weight_decay",
})
_SSM_ALLOWED_KEYS: frozenset = frozenset({
    "batch_size", "max_epochs", "lr", "weight_decay", "d_state",
})
_BAE_ALLOWED_KEYS: frozenset = frozenset({
    "batch_size", "max_epochs", "lr", "weight_decay", "n_estimators", "latent_dim",
})

_XGBOOST_PARAM_TYPES: dict[str, type] = {
    "num_boost_round": int, "objective": str, "eval_metric": list,
    "booster": str, "tree_method": str, "verbosity": int,
    "eta": float, "max_depth": int, "min_child_weight": int,
    "subsample": float, "lambda": float, "alpha": float,
}
_PYTORCH_PARAM_TYPES: dict[str, type] = {
    "batch_size": int, "max_epochs": int, "lr": float, "weight_decay": float,
}
_SSM_PARAM_TYPES: dict[str, type] = {
    "batch_size": int, "max_epochs": int, "lr": float, "weight_decay": float,
    "d_state": int,
}
_BAE_PARAM_TYPES: dict[str, type] = {
    "batch_size": int, "max_epochs": int, "lr": float, "weight_decay": float,
    "n_estimators": int, "latent_dim": int,
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


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


def _load_static_config() -> dict[str, Any]:
    """Load k3s/config.yaml for MLflow/serving defaults."""
    path = Path(_REPO_CONFIG_PATH)
    if path.exists():
        with open(path) as fh:
            return _yaml.safe_load(fh) or {}
    return {}


# ─── Models ──────────────────────────────────────────────────────────────────


class TuningConfig(BaseModel):
    enabled: bool = True
    number_of_trials: int = Field(3, ge=1)


class ModelConfig(BaseModel):
    experiment_name: str = "kuberay-attack-detection"
    registry_model_name: str = "attack-detection"
    mlflow_tracking_uri: str = ""
    mlflow_artifact_location: str = ""
    target: str = "attack"
    num_classes: int = Field(6, ge=2)
    seed: int = 42
    task_type: Literal["classification", "regression"] = "classification"
    model_type: str = "mlp"


class ResourceConstraintsConfig(BaseModel):
    """GPU resource constraints forwarded to GPUSelectorService in the Airflow DAG."""
    class GPUFallbackEntryConfig(BaseModel):
        infra: str
        accelerators: str
        use_spot: bool = False

    providers: list[str] = Field(default_factory=lambda: ["runpod"])
    gpu_types: list[str] | None = None
    min_vram_gb: float = 0
    max_price_per_hour: float = 9999
    prefer_spot: bool = True
    require_infiniband: bool = False
    preferred_regions: list[str] = Field(default_factory=list)
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    job_type: str = "tabular"
    gpu_fallbacks: list[GPUFallbackEntryConfig] | None = None


class RunRequest(BaseModel):
    preprocess_run_id: str            # references the preprocessing run; replaces processed_table
    execution_id: str = ""            # auto-generated if empty; becomes train_run_id
    framework: Literal["xgboost", "pytorch", "ssm", "bae"] = "xgboost"
    use_gpu: bool = False
    use_managed_jobs: bool = False    # SkyPilot mode: jobs.launch vs launch
    num_nodes: int = Field(1, ge=1, le=64)                        # Phase 5: multi-node training
    resource_constraints: ResourceConstraintsConfig | None = None  # Phase 3: dynamic GPU selection
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sample_fraction_for_tuning: float = Field(0.2, ge=0.01, le=1.0)
    hyperparams: dict[str, Any] = {}
    tune_settings: dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_hyperparams(self) -> "RunRequest":
        """Validate hyperparameter keys and types against the allowed set."""
        fw = self.framework
        model_type = self.model.model_type if self.model else "mlp"

        if fw == "xgboost":
            allowed = _XGBOOST_ALLOWED_KEYS
            expected_types = _XGBOOST_PARAM_TYPES
        elif model_type == "ssm":
            allowed = _SSM_ALLOWED_KEYS
            expected_types = _SSM_PARAM_TYPES
        elif model_type == "bae":
            allowed = _BAE_ALLOWED_KEYS
            expected_types = _BAE_PARAM_TYPES
        else:
            allowed = _PYTORCH_ALLOWED_KEYS
            expected_types = _PYTORCH_PARAM_TYPES

        for key, val in self.hyperparams.items():
            if key not in allowed:
                raise ValueError(
                    f"hyperparams['{key}'] is not a valid {fw} parameter. "
                    f"Allowed keys: {sorted(allowed)}"
                )
            exp_type = expected_types.get(key)
            if exp_type is not None and not isinstance(val, exp_type):
                if exp_type is float and isinstance(val, int):
                    pass
                else:
                    raise ValueError(
                        f"hyperparams['{key}'] expected {exp_type.__name__}, "
                        f"got {type(val).__name__}"
                    )

        return self


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _generate_train_run_id(dataset: str, preprocess_run_id: str) -> str:
    """Generate a unique, typed training run ID that embeds the preprocess suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    pre_suffix = preprocess_run_id[-6:]
    rand = secrets.token_hex(3)
    return f"train-{dataset}-{pre_suffix}-{ts}Z-{rand}"


def _fetch_preprocess_params(preprocess_run_id: str, s3, bucket: str) -> dict:
    """Load params_preprocess.yaml for the given run ID from S3.

    Tries new path first, then legacy path for historical runs.
    """
    def _get(key: str) -> str | None:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    yaml_content = _get(f"runs/preprocessing/{preprocess_run_id}/params_preprocess.yaml")
    if yaml_content is None:
        yaml_content = _get(f"params/{preprocess_run_id}/params_preprocess.yaml")
    if yaml_content is None:
        raise HTTPException(
            status_code=404,
            detail=f"params_preprocess.yaml not found for preprocess_run_id '{preprocess_run_id}'. "
                   "Run a preprocessing job first.",
        )
    return _yaml.safe_load(yaml_content) or {}


def _airflow_request(method: str, path: str, body=None):
    """Make an Airflow API call via the k8s client."""
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

    token_resp = _proxy("POST", "auth/token", b={"username": AIRFLOW_USER, "password": AIRFLOW_PASSWORD})
    token = _json.loads(token_resp.data).get("access_token", "")

    http_resp = _proxy(method, path, b=body, extra_headers={"Authorization": f"Bearer {token}"})
    try:
        data = _json.loads(http_resp.data)
    except (ValueError, AttributeError):
        data = {"raw": (http_resp.data or b"").decode("utf-8", errors="replace")}
    return http_resp.status, data


def _generate_training_params_yaml(
    req: RunRequest,
    train_run_id: str,
    lineage: dict,
    cfg: dict,
) -> str:
    """Build params_training.yaml with lineage block and full training config."""
    mlflow_cfg = cfg.get("mlflow", {})
    default_tracking_uri = (
        (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
        or str(mlflow_cfg.get("tracking_uri", "")).strip()
        or "http://my-mlflow"
    )
    default_artifact_location = (
        str(mlflow_cfg.get("artifact_base", "")).strip()
        or f"s3://{S3_BUCKET}/mlflow-artifacts/"
    )
    mlflow_tracking_uri = (req.model.mlflow_tracking_uri or "").strip() or default_tracking_uri
    mlflow_artifact_location = (
        (req.model.mlflow_artifact_location or "").strip()
        or default_artifact_location
    )

    tune_defaults = (
        _XGBOOST_TUNE_SETTINGS_DEFAULTS if req.framework == "xgboost"
        else _PYTORCH_TUNE_SETTINGS_DEFAULTS
    )
    resolved_tune_settings = {**tune_defaults, **req.tune_settings}

    hyperparams_block: dict[str, Any] = {req.framework: req.hyperparams}
    if req.tuning.enabled:
        hyperparams_block["tuning"] = resolved_tune_settings

    params: dict[str, Any] = {
        "run_metadata": {
            "run_id": train_run_id,
            "run_type": "training",
            "dataset": lineage.get("dataset", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "lineage": lineage,
        "execution": {
            "train_run_id": train_run_id,
            "processed_table": lineage["processed_table"],
            "tuning": {
                "enabled": req.tuning.enabled,
                "number_of_trials": req.tuning.number_of_trials,
            },
        },
        "splits": lineage.get("splits", {}),
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
        "kuberay": {
            "model": {
                "tune": req.tuning.enabled,
                "sample_fraction_for_tuning": req.sample_fraction_for_tuning,
                "target": req.model.target,
                "num_classes": req.model.num_classes,
                "dsl_count_dim": True,
                "input_dim": 14,
                "framework": req.framework,
                "task_type": req.model.task_type,
                "model_type": req.model.model_type,
                "use_gpu": req.use_gpu,
                "seed": req.model.seed,
                "mlflow_tracking_uri": mlflow_tracking_uri,
                "mlflow_experiment_name": req.model.experiment_name,
                "mlflow_artifact_location": mlflow_artifact_location,
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


def _build_lineage(preprocess_run_id: str, preprocess_params: dict, bucket: str) -> dict:
    """Build the lineage block for params_training.yaml from preprocessing params."""
    exec_cfg = preprocess_params.get("execution", {})
    dataset = exec_cfg.get("raw_dataset_name", "")
    artifact_set_id = preprocess_run_id[-6:]
    processed_table = f"iceberg.processed.{dataset}_{artifact_set_id}" if dataset else ""

    return {
        "preprocess_run_id": preprocess_run_id,
        "preprocess_params_s3_path": (
            f"s3://{bucket}/runs/preprocessing/{preprocess_run_id}/params_preprocess.yaml"
        ),
        "processed_table": processed_table,
        "dsl_s3_path": exec_cfg.get("dsl_s3_path", ""),
        "dsl_version": exec_cfg.get("dsl_version", ""),
        "schema_version": preprocess_params.get("schema_ref", {}).get("version", ""),
        "dataset": dataset,
        "splits": preprocess_params.get("splits", {}),
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/ids")
async def list_training_run_ids():
    """List training run IDs from S3 (new-style runs in runs/training/)."""
    import re as _re
    s3 = _s3_client()
    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="runs/training/",
        Delimiter="/",
    )
    runs = []
    for prefix_obj in resp.get("CommonPrefixes", []):
        run_id = prefix_obj["Prefix"].rstrip("/").rsplit("/", 1)[-1]
        m = _re.match(r"^train-(.+)-[0-9a-f]{6}-(\d{8}T\d{6}Z)-([0-9a-f]{6})$", run_id)
        dataset = m.group(1) if m else ""
        runs.append({"train_run_id": run_id, "dataset": dataset})
    # Newest first (run IDs contain timestamps)
    runs.sort(key=lambda r: r["train_run_id"], reverse=True)
    return {"runs": runs}


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
    """Generate params_training.yaml, upload to S3, and trigger training_pipeline DAG."""
    s3 = _s3_client()

    # Fetch preprocessing params to build lineage
    preprocess_params = _fetch_preprocess_params(request.preprocess_run_id, s3, S3_BUCKET)
    lineage = _build_lineage(request.preprocess_run_id, preprocess_params, S3_BUCKET)

    dataset = lineage["dataset"]
    train_run_id = (
        request.execution_id.strip()
        or _generate_train_run_id(dataset, request.preprocess_run_id)
    )

    cfg = _load_static_config()
    training_yaml_str = _generate_training_params_yaml(request, train_run_id, lineage, cfg)

    # Write ONLY to new path: runs/training/{train_run_id}/
    training_key = f"runs/training/{train_run_id}/params_training.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=training_key,
        Body=training_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    train_params_s3_path = f"s3://{S3_BUCKET}/{training_key}"

    # Route to SkyPilot DAG when GPU or explicit resource_constraints requested
    use_skypilot = request.use_gpu or request.resource_constraints is not None
    active_dag_id = DAG_ID_SKYPILOT if use_skypilot else DAG_ID

    # Trigger Airflow training DAG
    dag_conf: dict[str, Any] = {
        "train_run_id": train_run_id,
        "preprocess_run_id": request.preprocess_run_id,
        "dataset": dataset,
        "processed_table": lineage["processed_table"],
        "dsl_s3_path": lineage["dsl_s3_path"],
        "train_params_s3_path": train_params_s3_path,
        "model_type": request.framework,
        "num_nodes": request.num_nodes,
    }
    if use_skypilot:
        dag_conf["use_managed_jobs"] = bool(request.use_managed_jobs)

    if request.resource_constraints is not None:
        rc_dict = request.resource_constraints.model_dump()
        any_of = build_any_of_from_constraints(rc_dict)

        if has_explicit_resource_selection(rc_dict) and not any_of:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No valid SkyPilot resource entries could be generated from the selected "
                    "GPUs/regions. Adjust gpu_types/preferred_regions or provide gpu_fallbacks."
                ),
            )

        if any_of:
            rc_dict["gpu_fallbacks"] = any_of

        dag_conf["resource_constraints"] = rc_dict

    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{active_dag_id}/dagRuns",
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
        "dag_id": active_dag_id,
        "train_run_id": train_run_id,
        "preprocess_run_id": request.preprocess_run_id,
        "train_params_s3_path": train_params_s3_path,
        "processed_table": lineage["processed_table"],
        "skypilot": use_skypilot,
        "use_managed_jobs": bool(request.use_managed_jobs) if use_skypilot else False,
    }


@router.get("/{dag_run_id}/status")
async def run_status(dag_run_id: str, skypilot: bool = False):
    """Poll the state of an Airflow DAG run.

    Pass ?skypilot=true when the run was submitted to training_pipeline_skypilot.
    """
    dag_id = DAG_ID_SKYPILOT if skypilot else DAG_ID
    try:
        status, data = _airflow_request(
            "GET", f"api/v2/dags/{dag_id}/dagRuns/{dag_run_id}"
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
