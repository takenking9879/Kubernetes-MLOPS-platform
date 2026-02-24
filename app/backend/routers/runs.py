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
from pydantic import BaseModel

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


class RunRequest(BaseModel):
    dataset: str                                       # e.g. "network_traffic"
    dsl_version: int                                   # version number from /dsls listing
    execution_id: str = ""                             # auto-generated if empty
    framework: Literal["xgboost", "pytorch"] = "xgboost"
    splits: Splits
    hyperparams: dict[str, Any] = {}                  # optional overrides per framework


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

    params: dict[str, Any] = {
        "execution": {
            "execution_id": execution_id,
            "raw_table": f"iceberg.raw.{req.dataset}",
            "dsl_s3_path": _resolve_dsl_s3_path(req.dataset, req.dsl_version),
        },
        "splits": {
            "train": {"start": req.splits.train.start, "end": req.splits.train.end},
            "val":   {"start": req.splits.val.start,   "end": req.splits.val.end},
            "test":  {"start": req.splits.test.start,  "end": req.splits.test.end},
        },
        "model": {
            "framework": req.framework,
            "experiment_name": "kuberay-attack-detection",
            "registry_model_name": "attack-detection",
            "target": "attack",
            "num_classes": 6,
            "tune": True,
            "sample_fraction_for_tuning": 0.2,
            "seed": 42,
        },
        "hyperparams": {
            req.framework: req.hyperparams,
        },
        # ── Backward-compat block (serving layer reads kuberay.*) ──────────
        "kuberay": {
            "model": {
                "tune": True,
                "sample_fraction_for_tuning": 0.2,
                "target": "attack",
                "num_classes": 6,
                "dsl_count_dim": True,
                "input_dim": 14,
                "framework": req.framework,
                "seed": 42,
                "mlflow_tracking_uri": mlflow_cfg.get("tracking_uri", "http://my-mlflow"),
                "mlflow_experiment_name": "kuberay-attack-detection",
                "mlflow_artifact_location": mlflow_cfg.get("artifact_base", ""),
                "mlflow_registry_model_name": "attack-detection",
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
            "num_classes": 6,
            "target": "attack",
            "schemas": {
                "input_schema": "kafka_schema_features",
                "features_schema": "schema_features",
                "output_schema": "prediction_schema",
                "preprocessed_schema": "schema_preprocessed",
                "full_schema": "schema_full",
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
