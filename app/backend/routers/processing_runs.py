"""
Processing pipeline router.

Endpoints:
  POST /api/v2/processing-runs                 - trigger Spark-only preprocessing DAG
  GET  /api/v2/processing-runs                 - list processed tables from Iceberg metadata
  GET  /api/v2/processing-runs/{dag_run_id}/status - poll Airflow DAG run state
  GET  /api/v2/processing-runs/{run_id}/params - fetch params_preprocess.yaml for a run
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import boto3
import requests as _requests
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/api/v2/processing-runs", tags=["processing-runs"])

# ─── Config ──────────────────────────────────────────────────────────────────

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AIRFLOW_SERVICE = os.getenv("AIRFLOW_SERVICE", "my-airflow-api-server")
AIRFLOW_NAMESPACE = os.getenv("AIRFLOW_NAMESPACE", "airflow")
AIRFLOW_PORT = os.getenv("AIRFLOW_PORT", "8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "preprocessing_pipeline"

_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


# ─── Models ──────────────────────────────────────────────────────────────────


class SplitRange(BaseModel):
    start: str
    end: str


class Splits(BaseModel):
    train: SplitRange
    val: SplitRange
    test: SplitRange


class ProcessingRunRequest(BaseModel):
    dataset: str
    dsl_version: int = Field(..., ge=1)
    execution_id: str = ""          # auto-generated if empty
    splits: Splits

    @model_validator(mode="after")
    def validate_splits(self) -> "ProcessingRunRequest":
        """Validate split dates: format, positive duration, no pairwise overlap."""

        def _parse(s: str, label: str) -> datetime:
            try:
                return datetime.strptime(s, _DATE_FMT)
            except ValueError:
                raise ValueError(f"{label} must be YYYY-MM-DD HH:MM:SS, got '{s}'")

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
        for name, (s, e) in parsed.items():
            if s >= e:
                raise ValueError(f"splits.{name}.start must be before splits.{name}.end")
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        overlapping = []
        for a, b in pairs:
            a_s, a_e = parsed[a]
            b_s, b_e = parsed[b]
            if a_s < b_e and b_s < a_e:
                overlapping.append(f"{a} and {b}")
        if overlapping:
            raise ValueError(f"Splits overlap — conflicting pairs: {', '.join(overlapping)}")
        return self


class ProcessedTableEntry(BaseModel):
    execution_id: str
    dataset: str           # alias de raw_dataset_name (backward-compat)
    processed_table_name: str
    pipeline_hash: str = ""
    created_at: str = ""
    raw_dataset_name: str = ""  # nombre explícito del dataset fuente
    dsl_name: str = ""         # e.g. "v1__network_traffic.yaml"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _generate_preprocess_run_id(dataset: str) -> str:
    """Generate a unique, typed preprocess run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = secrets.token_hex(3)
    return f"pre-{dataset}-{ts}Z-{rand}"


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


def _get_latest_schema_version(dataset: str, s3_client, bucket: str) -> int:
    """List schemas/datasets/{dataset}/v*/ prefixes and return the highest version. 0 if none."""
    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"schemas/datasets/{dataset}/v",
        Delimiter="/",
    )
    versions = []
    for prefix_obj in resp.get("CommonPrefixes", []):
        match = re.search(r"/v(\d+)/$", prefix_obj["Prefix"])
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0)


def _build_schema_ref(dataset: str, bucket: str, s3_client) -> dict:
    """Build schema_ref using the latest available schema version.

    Auto-detects version from S3. Never derives paths from table names.
    Raises 404 if no schema exists for the dataset.
    """
    version = _get_latest_schema_version(dataset, s3_client, bucket)
    if version == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No schema found for dataset '{dataset}'. "
                "Define and save full.yaml on the Processing page first."
            ),
        )
    base = f"s3://{bucket}/schemas/datasets/{dataset}/v{version}"
    return {"version": version, "full": f"{base}/full.yaml"}


def _validate_schema_ref_exists(schema_ref: dict, s3_client, bucket: str) -> None:
    """Head-object check for each schema URI. Raises HTTPException(404) if missing."""
    for key, uri in schema_ref.items():
        if key == "version":
            continue
        parsed = urlparse(uri)
        try:
            s3_client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise HTTPException(
                    status_code=404,
                    detail=f"Schema '{key}' not found at {uri}. Define it on the Processing page.",
                )
            raise


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def submit_processing_run(request: ProcessingRunRequest):
    """Trigger the Airflow preprocessing_pipeline DAG."""
    preprocess_run_id = (
        request.execution_id.strip()
        or _generate_preprocess_run_id(request.dataset)
    )

    dsl_s3_path = _resolve_dsl_s3_path(request.dataset, request.dsl_version)

    s3 = _s3_client()

    # Build schema_ref (auto-detects latest version from S3)
    schema_ref = _build_schema_ref(request.dataset, S3_BUCKET, s3)
    _validate_schema_ref_exists(schema_ref, s3, S3_BUCKET)

    # artifact_set_id for Spark: last 6 chars of the run ID
    artifact_set_id = preprocess_run_id[-6:]

    import yaml as _yaml

    params: dict[str, Any] = {
        "run_metadata": {
            "run_id": preprocess_run_id,
            "run_type": "preprocessing",
            "dataset": request.dataset,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "execution": {
            "preprocess_run_id": preprocess_run_id,
            "raw_table": f"iceberg.raw.{request.dataset}",
            "raw_dataset_name": request.dataset,
            "dsl_s3_path": dsl_s3_path,
            "dsl_version": request.dsl_version,
        },
        "schema_ref": schema_ref,
        "splits": {
            "train": {"start": request.splits.train.start, "end": request.splits.train.end},
            "val":   {"start": request.splits.val.start,   "end": request.splits.val.end},
            "test":  {"start": request.splits.test.start,  "end": request.splits.test.end},
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
                "table_prefix": request.dataset,
            },
        },
    }
    params_yaml_str = _yaml.dump(params, default_flow_style=False, allow_unicode=True)

    # Write ONLY to new path: runs/preprocessing/{run_id}/
    params_key = f"runs/preprocessing/{preprocess_run_id}/params_preprocess.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=params_key,
        Body=params_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    preprocess_params_s3_path = f"s3://{S3_BUCKET}/{params_key}"

    # Trigger Airflow preprocessing_pipeline DAG
    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{DAG_ID}/dagRuns",
            body={
                "logical_date": datetime.now(timezone.utc).isoformat(),
                "conf": {
                    "preprocess_run_id": preprocess_run_id,
                    "artifact_set_id": artifact_set_id,
                    "dataset": request.dataset,
                    "raw_table": f"iceberg.raw.{request.dataset}",
                    "dsl_s3_path": dsl_s3_path,
                    "preprocess_params_s3_path": preprocess_params_s3_path,
                    "schema_s3_path": schema_ref["full"],
                },
            },
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
        "preprocess_run_id": preprocess_run_id,
        "artifact_set_id": artifact_set_id,
        "preprocess_params_s3_path": preprocess_params_s3_path,
        "dsl_s3_path": dsl_s3_path,
        "schema_version": schema_ref["version"],
    }


@router.get("")
async def list_processing_runs(dataset: str | None = Query(None, description="Filtrar por raw dataset name")):
    """List available processed tables from iceberg.metadata.preprocessing_artifacts."""
    try:
        from pyiceberg.catalog import load_catalog

        catalog = load_catalog(
            "glue",
            **{
                "type": "glue",
                "client.region": AWS_REGION,
                "client.access-key-id": os.getenv("AWS_ACCESS_KEY_ID", ""),
                "client.secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            },
        )
        table = catalog.load_table("metadata.preprocessing_artifacts")
        # Scan all fields — raw_dataset_name may not exist on tables created before the migration.
        df = table.scan().to_pandas()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query Iceberg metadata table: {exc}",
        ) from exc

    runs = []
    for _, row in df.iterrows():
        table_name: str = str(row.get("processed_table_name", ""))

        # raw_dataset_name: read directly from column (post-migration)
        # Fallback: parse from table name for historical rows with NULL
        raw_dataset_name = str(row.get("raw_dataset_name") or "")
        if not raw_dataset_name and "." in table_name:
            tail = table_name.rsplit(".", 1)[-1]
            parts = tail.rsplit("_", 2)
            if len(parts) == 3:
                raw_dataset_name = parts[0]

        runs.append(
            ProcessedTableEntry(
                execution_id=str(row.get("artifact_set_id", "")),
                dataset=raw_dataset_name,
                processed_table_name=table_name,
                pipeline_hash=str(row.get("pipeline_hash", "")),
                created_at=str(row.get("created_at", "")),
                raw_dataset_name=raw_dataset_name,
                dsl_name=str(row.get("dsl_name") or ""),
            )
        )

    # Filter by dataset if specified
    if dataset:
        runs = [r for r in runs if r.raw_dataset_name == dataset]

    # Most recent first
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return {"runs": [r.model_dump() for r in runs]}


@router.get("/ids")
async def list_preprocess_run_ids(dataset: str | None = Query(default=None)):
    """List preprocessing run IDs from S3 (new-style runs in runs/preprocessing/)."""
    s3 = _s3_client()
    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="runs/preprocessing/",
        Delimiter="/",
    )
    runs = []
    for prefix_obj in resp.get("CommonPrefixes", []):
        run_id = prefix_obj["Prefix"].rstrip("/").rsplit("/", 1)[-1]
        m = re.match(r"^pre-(.+)-(\d{8}T\d{6}Z)-([0-9a-f]{6})$", run_id)
        run_dataset = m.group(1) if m else ""
        runs.append({"preprocess_run_id": run_id, "dataset": run_dataset})
    if dataset:
        runs = [r for r in runs if r["dataset"] == dataset]
    # Newest first (run IDs contain timestamps)
    runs.sort(key=lambda r: r["preprocess_run_id"], reverse=True)
    return {"runs": runs}


@router.get("/{run_id}/params")
async def get_preprocess_params(run_id: str):
    """Fetch params_preprocess.yaml for a given run_id from S3.

    Tries new path first (runs/preprocessing/{run_id}/), then legacy path
    (params/{run_id}/) for historical runs.
    """
    s3 = _s3_client()

    def _fetch_key(key: str) -> str | None:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            return obj["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise HTTPException(status_code=500, detail=f"S3 error: {e}") from e

    # New path first
    yaml_content = _fetch_key(f"runs/preprocessing/{run_id}/params_preprocess.yaml")
    if yaml_content is None:
        # Legacy fallback for historical runs (read-only — new runs never write here)
        yaml_content = _fetch_key(f"params/{run_id}/params_preprocess.yaml")
    if yaml_content is None:
        yaml_content = _fetch_key(f"params/{run_id}/params.yaml")
    if yaml_content is None:
        raise HTTPException(
            status_code=404,
            detail=f"No params file found for run_id '{run_id}'",
        )
    return {"run_id": run_id, "yaml_content": yaml_content}


@router.get("/{dag_run_id}/status")
async def processing_run_status(dag_run_id: str):
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
