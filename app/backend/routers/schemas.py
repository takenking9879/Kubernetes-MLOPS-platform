"""
Schema upload router.

Endpoints:
  POST /api/v2/schemas/{dataset}/full  — upload full.yaml for a dataset (auto-version)
  POST /api/v2/schemas/{dataset}/raw   — upload raw.yaml for a dataset (auto-version)

Versions are tracked independently: uploading full.yaml does NOT affect the raw.yaml
version counter and vice versa. Starting from v1 if no prior version exists.
"""

from __future__ import annotations

import os
import re

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2/schemas", tags=["schemas"])

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


class SingleSchemaUpload(BaseModel):
    yaml_content: str


def _next_version(dataset: str, schema_type: str, s3, bucket: str) -> int:
    """Scan schemas/datasets/{dataset}/v*/{schema_type}.yaml, return max_version + 1 (min 1)."""
    prefix = f"schemas/datasets/{dataset}/v"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    versions = []
    for obj in resp.get("Contents", []):
        m = re.search(rf"/v(\d+)/{re.escape(schema_type)}\.yaml$", obj["Key"])
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=0) + 1


@router.post("/{dataset}/full", status_code=201)
async def upload_full_schema(dataset: str, body: SingleSchemaUpload):
    """Upload full.yaml for a dataset, auto-incrementing the version."""
    s3 = _s3_client()
    version = _next_version(dataset, "full", s3, S3_BUCKET)
    key = f"schemas/datasets/{dataset}/v{version}/full.yaml"
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body.yaml_content.encode("utf-8"),
            ContentType="application/x-yaml",
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}") from e
    return {"version": version, "s3_path": f"s3://{S3_BUCKET}/{key}"}


@router.post("/{dataset}/raw", status_code=201)
async def upload_raw_schema(dataset: str, body: SingleSchemaUpload):
    """Upload raw.yaml for a dataset, auto-incrementing the version independently of full.yaml."""
    s3 = _s3_client()
    version = _next_version(dataset, "raw", s3, S3_BUCKET)
    key = f"schemas/datasets/{dataset}/v{version}/raw.yaml"
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body.yaml_content.encode("utf-8"),
            ContentType="application/x-yaml",
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}") from e
    return {"version": version, "s3_path": f"s3://{S3_BUCKET}/{key}"}
