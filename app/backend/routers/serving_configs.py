"""
Serving configuration router.

Endpoints:
  POST /api/v2/serving-configs             — generate params_serving.yaml, upload to S3
  GET  /api/v2/serving-configs/{id}/params — fetch params_serving.yaml for a run

The caller (ServingPage) must save raw.yaml first via POST /api/v2/schemas/{dataset}/raw,
then pass the returned s3_path as raw_schema_s3_path in the ServingConfigRequest.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any

import boto3
import yaml as _yaml
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2/serving-configs", tags=["serving-configs"])

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


# ─── Models ──────────────────────────────────────────────────────────────────


class ServingConfigRequest(BaseModel):
    train_run_id: str
    raw_schema_s3_path: str               # saved via POST /api/v2/schemas/{dataset}/raw
    alias: str = "champion"
    canary: bool = False
    canary_alias: str = "challenger"
    canary_probability: float = Field(0.10, ge=0.0, le=1.0)
    initial_replicas: int = Field(0, ge=0)
    webhook_public_base_url: str = ""
    webhook_path: str = "/infer/webhook"
    webhook_max_timestamp_age_seconds: int = Field(300, ge=1)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _generate_serve_run_id(dataset: str, train_run_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    train_suffix = train_run_id[-6:]
    rand = secrets.token_hex(3)
    return f"serve-{dataset}-{train_suffix}-{ts}Z-{rand}"


def _fetch_training_params(train_run_id: str, s3, bucket: str) -> dict:
    """Load params_training.yaml for the given train_run_id from S3."""
    key = f"runs/training/{train_run_id}/params_training.yaml"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return _yaml.safe_load(obj["Body"].read().decode("utf-8")) or {}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=404,
                detail=f"params_training.yaml not found for train_run_id '{train_run_id}'.",
            ) from e
        raise


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def submit_serving_config(request: ServingConfigRequest):
    """Generate params_serving.yaml and save to s3://bucket/runs/serving/{serve_run_id}/."""
    s3 = _s3_client()

    # Fetch training params to extract lineage (dataset, preprocess_run_id, etc.)
    training_params = _fetch_training_params(request.train_run_id, s3, S3_BUCKET)
    lineage = training_params.get("lineage", {})
    dataset = lineage.get("dataset", "")
    preprocess_run_id = lineage.get("preprocess_run_id", "")

    if not dataset:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve dataset from training params lineage block.",
        )

    serve_run_id = _generate_serve_run_id(dataset, request.train_run_id)

    params: dict[str, Any] = {
        "run_metadata": {
            "serve_run_id": serve_run_id,
            "train_run_id": request.train_run_id,
            "preprocess_run_id": preprocess_run_id,
            "run_type": "serving",
            "dataset": dataset,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "lineage": {
            "preprocess_run_id": preprocess_run_id,
            "train_run_id": request.train_run_id,
        },
        "kafka_inference": {
            "raw_schema_s3_path": request.raw_schema_s3_path,
        },
        "serving": {
            "alias": request.alias,
            "canary": request.canary,
            "webhook_public_base_url": request.webhook_public_base_url,
            "webhook_path": request.webhook_path,
            "webhook_max_timestamp_age_seconds": request.webhook_max_timestamp_age_seconds,
        },
        "canary": {
            "alias": request.canary_alias,
            "canary_probability": request.canary_probability,
            "initial_replicas": request.initial_replicas,
        },
    }

    params_yaml_str = _yaml.dump(params, default_flow_style=False, allow_unicode=True)
    key = f"runs/serving/{serve_run_id}/params_serving.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=params_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )

    return {
        "serve_run_id": serve_run_id,
        "params_s3_path": f"s3://{S3_BUCKET}/{key}",
        "dataset": dataset,
        "train_run_id": request.train_run_id,
    }


@router.get("/{serve_run_id}/params")
async def get_serving_params(serve_run_id: str):
    """Fetch params_serving.yaml for a given serve_run_id."""
    s3 = _s3_client()
    key = f"runs/serving/{serve_run_id}/params_serving.yaml"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        yaml_content = obj["Body"].read().decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=404,
                detail=f"No params_serving.yaml found for serve_run_id '{serve_run_id}'",
            ) from e
        raise
    return {"serve_run_id": serve_run_id, "yaml_content": yaml_content}
