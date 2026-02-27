"""
Processing pipeline router.

Endpoints:
  POST /api/v2/processing-runs                 - trigger Spark-only preprocessing DAG
  GET  /api/v2/processing-runs                 - list processed tables from Iceberg metadata
  GET  /api/v2/processing-runs/{dag_run_id}/status - poll Airflow DAG run state
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import boto3
import requests as _requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2/processing-runs", tags=["processing-runs"])

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

_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


# ─── Models ──────────────────────────────────────────────────────────────────


class ProcessingRunRequest(BaseModel):
    dataset: str
    dsl_version: int = Field(..., ge=1)
    execution_id: str = ""          # auto-generated if empty


class ProcessedTableEntry(BaseModel):
    execution_id: str
    dataset: str
    processed_table_name: str
    pipeline_hash: str = ""
    created_at: str = ""


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


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def submit_processing_run(request: ProcessingRunRequest):
    """Trigger the Airflow ml_pipeline DAG in preprocessing_only mode."""
    execution_id = request.execution_id.strip() or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    dsl_s3_path = _resolve_dsl_s3_path(request.dataset, request.dsl_version)

    # Upload a minimal params.yaml so the Spark task can read execution config
    import yaml as _yaml

    params: dict[str, Any] = {
        "execution": {
            "execution_id": execution_id,
            "raw_table": f"iceberg.raw.{request.dataset}",
            "dsl_s3_path": dsl_s3_path,
        },
        "iceberg_tables": {
            "warehouse": f"s3://{S3_BUCKET}/warehouse",
            "raw": {
                "catalog": "iceberg",
                "namespace": "raw",
                "table": request.dataset,
                "full_name": f"iceberg.raw.{request.dataset}",
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
            },
        },
    }
    params_yaml_str = _yaml.dump(params, default_flow_style=False, allow_unicode=True)

    s3 = _s3_client()
    params_key = f"params/{execution_id}/params.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=params_key,
        Body=params_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    params_s3_path = f"s3://{S3_BUCKET}/{params_key}"

    # Trigger Airflow DAG with preprocessing_only mode
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
                    "pipeline_mode": "preprocessing_only",
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


@router.get("")
async def list_processing_runs():
    """List available processed tables from iceberg.metadata.preprocessing_artifacts."""
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
            selected_fields=(
                "artifact_set_id",
                "processed_table_name",
                "pipeline_hash",
                "created_at",
            ),
        ).to_pandas()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query Iceberg metadata table: {exc}",
        ) from exc

    runs = []
    for _, row in df.iterrows():
        table_name: str = str(row.get("processed_table_name", ""))
        # Extract dataset from table name: iceberg.processed.{dataset}_{execution_id}
        # Convention: last part after "processed." is "{dataset}_{execution_id}"
        dataset = ""
        if "." in table_name:
            tail = table_name.rsplit(".", 1)[-1]
            # execution_id format is YYYYMMDD_HHMMSS (15 chars including underscore)
            # dataset is everything before the last two underscore-separated segments
            parts = tail.rsplit("_", 2)
            if len(parts) == 3:
                dataset = parts[0]

        runs.append(
            ProcessedTableEntry(
                execution_id=str(row.get("artifact_set_id", "")),
                dataset=dataset,
                processed_table_name=table_name,
                pipeline_hash=str(row.get("pipeline_hash", "")),
                created_at=str(row.get("created_at", "")),
            )
        )

    # Most recent first
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return {"runs": [r.model_dump() for r in runs]}


@router.get("/{dag_run_id}/status")
async def processing_run_status(dag_run_id: str):
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
