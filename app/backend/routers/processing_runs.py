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


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def submit_processing_run(request: ProcessingRunRequest):
    """Trigger the Airflow ml_pipeline DAG in preprocessing_only mode."""
    execution_id = request.execution_id.strip() or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    dsl_s3_path = _resolve_dsl_s3_path(request.dataset, request.dsl_version)

    # Upload params_preprocess.yaml: solo config de preprocesamiento (sin model/kuberay/serving)
    import yaml as _yaml

    params: dict[str, Any] = {
        "execution": {
            "execution_id": execution_id,
            "raw_table": f"iceberg.raw.{request.dataset}",
            "raw_dataset_name": request.dataset,
            "dsl_s3_path": dsl_s3_path,
        },
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
            },
        },
    }
    params_yaml_str = _yaml.dump(params, default_flow_style=False, allow_unicode=True)

    s3 = _s3_client()
    params_key = f"params/{execution_id}/params_preprocess.yaml"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=params_key,
        Body=params_yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    preprocess_params_s3_path = f"s3://{S3_BUCKET}/{params_key}"

    # Trigger Airflow DAG with preprocessing_only mode
    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{DAG_ID}/dagRuns",
            body={
                "logical_date": datetime.now(timezone.utc).isoformat(),
                "conf": {
                    "execution_id": execution_id,
                    "raw_table": f"iceberg.raw.{request.dataset}",
                    "dsl_s3_path": dsl_s3_path,
                    "preprocess_params_s3_path": preprocess_params_s3_path,
                    "pipeline_mode": "preprocessing_only",
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
        "execution_id": execution_id,
        "preprocess_params_s3_path": preprocess_params_s3_path,
        "dsl_s3_path": dsl_s3_path,
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

        # raw_dataset_name: leer directamente de la columna (post-migración)
        # Fallback: parsear del nombre de tabla para filas históricas con NULL
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

    # Filtro por dataset si se especifica
    if dataset:
        runs = [r for r in runs if r.raw_dataset_name == dataset]

    # Most recent first
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return {"runs": [r.model_dump() for r in runs]}


@router.get("/{execution_id}/params")
async def get_preprocess_params(execution_id: str):
    """Fetch params_preprocess.yaml for a given execution_id from S3.

    Backward-compat: si no existe params_preprocess.yaml, intenta con el antiguo params.yaml.
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

    yaml_content = _fetch_key(f"params/{execution_id}/params_preprocess.yaml")
    if yaml_content is None:
        # Backward-compat: monolithic params.yaml de runs anteriores a la separación
        yaml_content = _fetch_key(f"params/{execution_id}/params.yaml")
    if yaml_content is None:
        raise HTTPException(
            status_code=404,
            detail=f"No params file found for execution_id '{execution_id}'",
        )
    return {"execution_id": execution_id, "yaml_content": yaml_content}


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
