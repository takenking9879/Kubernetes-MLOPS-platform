"""
DSL versioning router.

DSL files are stored in S3 using the convention:
    dsl/dsl_{dataset}/v{N}__{slug}.yaml

Endpoints:
  GET  /api/v2/datasets/{name}/dsls          - list DSL versions for a dataset
  GET  /api/v2/datasets/{name}/dsls/{version} - fetch a specific DSL version
  POST /api/v2/datasets/{name}/dsls          - save a new DSL version to S3
"""

from __future__ import annotations

import os
import re
from typing import Any

import boto3
import yaml as _yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2/datasets", tags=["dsls"])

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

_DATASET_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}$")
# Matches: v{N}__{slug}.yaml
_DSL_FILENAME_RE = re.compile(r"^v(\d+)__(.+)\.yaml$")


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


def _validate_dataset_name(name: str):
    if not _DATASET_NAME_RE.match(name):
        raise HTTPException(status_code=422, detail=f"Invalid dataset name: {name!r}")


def _slugify(text: str) -> str:
    """Convert a descriptive name to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug[:60] or "dsl"


def _dsl_prefix(dataset: str) -> str:
    return f"dsl/dsl_{dataset}/"


def _list_dsl_objects(s3, dataset: str) -> list[dict[str, Any]]:
    """List all DSL YAML objects under dsl/dsl_{dataset}/."""
    prefix = _dsl_prefix(dataset)
    paginator = s3.get_paginator("list_objects_v2")
    items = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]
            m = _DSL_FILENAME_RE.match(filename)
            if not m:
                continue
            items.append({
                "version": int(m.group(1)),
                "slug": m.group(2),
                "key": key,
                "last_modified": obj["LastModified"].isoformat(),
            })
    items.sort(key=lambda x: x["version"])
    return items


# ─── Models ──────────────────────────────────────────────────────────────────


class SaveDSLRequest(BaseModel):
    name: str          # human-readable descriptive name (required)
    yaml_content: str  # full DSL YAML string


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/{name}/dsls")
async def list_dsls(name: str):
    """List all DSL versions for a dataset."""
    _validate_dataset_name(name)
    s3 = _s3_client()
    items = _list_dsl_objects(s3, name)
    return {"dataset": name, "dsls": items}


@router.get("/{name}/dsls/{version}")
async def get_dsl(name: str, version: int):
    """Fetch the YAML content of a specific DSL version."""
    _validate_dataset_name(name)
    s3 = _s3_client()
    items = _list_dsl_objects(s3, name)
    match = next((i for i in items if i["version"] == version), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"DSL version {version} not found for dataset '{name}'.",
        )
    obj = s3.get_object(Bucket=S3_BUCKET, Key=match["key"])
    yaml_content = obj["Body"].read().decode("utf-8")
    return {
        "version": match["version"],
        "slug": match["slug"],
        "key": match["key"],
        "yaml_content": yaml_content,
    }


@router.post("/{name}/dsls", status_code=201)
async def save_dsl(name: str, request: SaveDSLRequest):
    """Save a new versioned DSL to S3.

    - Validates that the DSL YAML is non-empty and has a 'pipeline' key.
    - Rejects duplicate descriptive names within the same dataset.
    - Auto-increments version (max existing + 1).
    """
    _validate_dataset_name(name)

    # Basic YAML validation
    if not request.yaml_content.strip():
        raise HTTPException(status_code=400, detail="yaml_content is empty.")
    try:
        parsed = _yaml.safe_load(request.yaml_content)
    except _yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict) or "pipeline" not in parsed:
        raise HTTPException(
            status_code=400,
            detail="DSL YAML must have a top-level 'pipeline' key.",
        )

    slug = _slugify(request.name)
    if not slug:
        raise HTTPException(status_code=400, detail="DSL name is required and must not be empty.")

    s3 = _s3_client()
    existing = _list_dsl_objects(s3, name)

    # Reject duplicate slug within this dataset
    existing_slugs = {i["slug"] for i in existing}
    if slug in existing_slugs:
        raise HTTPException(
            status_code=409,
            detail=f"A DSL named '{slug}' already exists for dataset '{name}'. "
                   "Choose a different name.",
        )

    next_version = max((i["version"] for i in existing), default=0) + 1
    key = f"{_dsl_prefix(name)}v{next_version}__{slug}.yaml"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=request.yaml_content.encode("utf-8"),
        ContentType="application/x-yaml",
    )

    return {"version": next_version, "slug": slug, "key": key}
