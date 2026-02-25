"""
Dataset management router.

Endpoints:
  GET  /api/v2/datasets                        - list datasets (S3 raw/ prefixes)
  POST /api/v2/datasets                        - create dataset (create S3 prefix)
  POST /api/v2/datasets/{name}/upload          - upload parquet files to S3 raw/{name}/
  POST /api/v2/datasets/{name}/ingest          - submit SparkApplication ingestion job
  GET  /api/v2/datasets/{name}/ingest/{job}/status - poll ingestion job state
  GET  /api/v2/datasets/{name}/sample          - fetch up to 3000 rows from iceberg.raw.{name}
  POST /api/v2/datasets/{name}/schemas         - upload versioned schema YAMLs to S3
"""

from __future__ import annotations

import copy
import os
import re
import time
from pathlib import Path
from typing import Any

import boto3
import yaml as _yaml
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2/datasets", tags=["datasets"])

# ─── Config ──────────────────────────────────────────────────────────────────

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
SPARK_NAMESPACE = os.getenv("SPARK_NAMESPACE", "spark")
SPARK_INGEST_YAML = os.getenv(
    "SPARK_INGEST_YAML",
    "/git-data/repo/k3s/spark/ingestion/spark-application-ingestion.yaml",
)

_DATASET_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}$")


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


def _validate_dataset_name(name: str):
    if not _DATASET_NAME_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid dataset name '{name}'. "
                "Must be lowercase, start with a letter, use only a-z / 0-9 / _, max 63 chars."
            ),
        )


# ─── Models ──────────────────────────────────────────────────────────────────


class CreateDatasetRequest(BaseModel):
    name: str


class DatasetInfo(BaseModel):
    name: str
    file_count: int
    size_bytes: int


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("", response_model=list[DatasetInfo])
async def list_datasets():
    """List all datasets as S3 prefixes under raw/."""
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    # List common prefixes under raw/ (one level deep)
    datasets: dict[str, DatasetInfo] = {}
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix="raw/", Delimiter="/")
    for page in pages:
        for prefix in page.get("CommonPrefixes", []):
            name = prefix["Prefix"].rstrip("/").split("/")[-1]
            datasets[name] = DatasetInfo(name=name, file_count=0, size_bytes=0)

    # Count files + total size for each dataset
    for name, info in datasets.items():
        file_pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=f"raw/{name}/")
        for fp in file_pages:
            for obj in fp.get("Contents", []):
                info.file_count += 1
                info.size_bytes += obj["Size"]

    return list(datasets.values())


@router.post("", response_model=DatasetInfo, status_code=201)
async def create_dataset(request: CreateDatasetRequest):
    """Create a new dataset by creating an S3 prefix raw/{name}/."""
    _validate_dataset_name(request.name)
    s3 = _s3_client()
    # Put an empty placeholder object to materialise the prefix
    s3.put_object(Bucket=S3_BUCKET, Key=f"raw/{request.name}/.keep", Body=b"")
    return DatasetInfo(name=request.name, file_count=0, size_bytes=0)


@router.post("/{name}/upload")
async def upload_files(name: str, files: list[UploadFile] = File(...)):
    """Upload one or more Parquet files to S3 raw/{name}/."""
    _validate_dataset_name(name)
    s3 = _s3_client()

    uploaded = []
    for f in files:
        filename = f.filename or "unknown.parquet"
        if not filename.lower().endswith(".parquet"):
            raise HTTPException(
                status_code=400,
                detail=f"Only Parquet files allowed. Got: {filename}",
            )
        content = await f.read()
        key = f"raw/{name}/{filename}"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content)
        uploaded.append({"key": key, "size_bytes": len(content)})

    return {"uploaded": uploaded}


@router.post("/{name}/ingest")
async def submit_ingest(name: str):
    """Submit a SparkApplication ingestion job for this dataset."""
    _validate_dataset_name(name)

    base_path = Path(SPARK_INGEST_YAML)
    if not base_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion YAML not found at {SPARK_INGEST_YAML}. "
                   "Set SPARK_INGEST_YAML env var.",
        )

    with open(base_path) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    ts = int(time.time())
    safe_name = name.replace("_", "-")
    job_name = f"spark-ingestion-{safe_name}-{ts}"
    manifest["metadata"]["name"] = job_name
    manifest["metadata"]["namespace"] = SPARK_NAMESPACE

    # Inject per-dataset env vars via sparkConf (same pattern as ml_pipeline_dag.py)
    raw_s3_path = f"s3://{S3_BUCKET}/raw/{name}/"
    spark_conf = manifest["spec"].setdefault("sparkConf", {})
    for key, val in {
        "ICEBERG_RAW_TABLE": name,
        "RAW_S3_PATH": raw_s3_path,
    }.items():
        spark_conf[f"spark.kubernetes.driverEnv.{key}"] = val
        spark_conf[f"spark.kubernetes.executorEnv.{key}"] = val

    try:
        from kubernetes import client as k8s_client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        custom = k8s_client.CustomObjectsApi()
        custom.create_namespaced_custom_object(
            group="spark.apache.org",
            version="v1",
            namespace=SPARK_NAMESPACE,
            plural="sparkapplications",
            body=manifest,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to submit SparkApplication: {exc}") from exc

    return {"job_name": job_name}


@router.get("/{name}/ingest/{job_name}/status")
async def ingest_status(name: str, job_name: str):
    """Poll the state of a SparkApplication ingestion job."""
    _validate_dataset_name(name)
    try:
        from kubernetes import client as k8s_client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        custom = k8s_client.CustomObjectsApi()
        obj = custom.get_namespaced_custom_object(
            group="spark.apache.org",
            version="v1",
            namespace=SPARK_NAMESPACE,
            plural="sparkapplications",
            name=job_name,
        )
        state = (
            obj.get("status", {})
               .get("currentState", {})
               .get("currentStateSummary", "Unknown")
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {exc}") from exc

    return {"state": state}


@router.get("/{name}/sample")
async def get_sample(name: str, limit: int = 3000):
    """Fetch up to `limit` rows from iceberg.raw.{name} via pyiceberg."""
    _validate_dataset_name(name)
    limit = min(limit, 3000)

    try:
        from pyiceberg.catalog.glue import GlueCatalog

        catalog = GlueCatalog(
            "glue",
            **{
                "glue.region": AWS_REGION,
                "s3.region": AWS_REGION,
                "warehouse": f"s3://{S3_BUCKET}/warehouse/",
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            },
        )
        table = catalog.load_table(f"raw.{name}")
        df = table.scan(limit=limit).to_pandas()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Iceberg sample for iceberg.raw.{name}: {exc}",
        ) from exc

    columns = [
        {"name": col, "sparkType": _pandas_to_spark(str(df[col].dtype)), "nullable": bool(df[col].isna().any())}
        for col in df.columns
    ]
    rows = df.head(limit).to_dict(orient="records")
    return {"columns": columns, "rows": rows, "row_count": len(rows)}


@router.get("/from-iceberg")
async def get_schema_from_iceberg(table: str):
    """Bridge for the DSL builder to load schema from any Iceberg table."""
    # table is expected as 'catalog.db.table' or just 'table' (which we treat as iceberg.raw.{table})
    # If it contains raw. we strip it to use get_sample
    name = table.split(".")[-1]
    res = await get_sample(name)
    return {
        "columns": res["columns"],
        "sampleRows": res["rows"][:10],
        "schemaHash": "iceberg",
        "uploadedPath": f"iceberg:{table}"
    }


class SchemaUploadRequest(BaseModel):
    raw: str            # YAML string for raw.yaml
    full: str           # YAML string for full.yaml
    preprocessed: str   # YAML string for preprocessed.yaml


@router.post("/{name}/schemas", status_code=201)
async def upload_schemas(name: str, request: SchemaUploadRequest):
    """Upload versioned schema YAMLs (raw, full, preprocessed) to S3.

    Schemas are stored under: schemas/datasets/{name}/v{N}/
    Version is auto-incremented by scanning existing S3 prefixes.
    """
    _validate_dataset_name(name)
    s3 = _s3_client()

    # Validate each YAML field: must parse and contain a top-level 'fields' list
    for field_name, content in [
        ("raw", request.raw),
        ("full", request.full),
        ("preprocessed", request.preprocessed),
    ]:
        try:
            parsed = _yaml.safe_load(content)
        except _yaml.YAMLError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid YAML in '{field_name}': {exc}",
            )
        if not isinstance(parsed, dict) or not isinstance(parsed.get("fields"), list):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Schema '{field_name}' must be a YAML dict with a top-level 'fields' list."
                ),
            )

    # Determine the next version number by scanning existing S3 prefixes
    prefix = f"schemas/datasets/{name}/v"
    paginator = s3.get_paginator("list_objects_v2")
    existing_versions: list[int] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # Key looks like: schemas/datasets/{name}/v3/
            segment = cp["Prefix"].rstrip("/").split("/")[-1]  # e.g. "v3"
            if segment.startswith("v") and segment[1:].isdigit():
                existing_versions.append(int(segment[1:]))

    next_version = (max(existing_versions) + 1) if existing_versions else 1

    # Upload each schema YAML to S3
    uploaded: dict[str, str] = {}
    for field_name, content in [
        ("raw", request.raw),
        ("full", request.full),
        ("preprocessed", request.preprocessed),
    ]:
        key = f"schemas/datasets/{name}/v{next_version}/{field_name}.yaml"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="application/x-yaml",
        )
        uploaded[field_name] = f"s3://{S3_BUCKET}/{key}"

    return {"version": next_version, "uploaded": uploaded}


_PANDAS_TYPE_MAP = {
    "int64": "long", "int32": "integer", "float64": "double",
    "float32": "double", "object": "string", "bool": "boolean",
    "datetime64": "timestamp",
}


def _pandas_to_spark(dtype_str: str) -> str:
    for k, v in _PANDAS_TYPE_MAP.items():
        if k in dtype_str:
            return v
    return "string"
