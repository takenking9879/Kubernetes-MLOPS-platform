"""
Serving configuration router.

Endpoints:
  POST /api/v2/serving-configs                              — generate params_serving.yaml, upload to S3
  GET  /api/v2/serving-configs/{id}/params                  — fetch params_serving.yaml for a run
    POST /api/v2/serving-configs/{id}/deploy                  — trigger tabular_serving_skypilot_pipeline Airflow DAG
  GET  /api/v2/serving-configs/{id}/deploy/{dag_run_id}/status — poll DAG run state
  POST /api/v2/serving-configs/{id}/update-champion         — rolling sky serve update to sync @champion (SkyPilot only)

Serving modes:
    ray_only — model metadata mode (SkyPilot deployment path)
    kafka    — legacy metadata mode; deployment path is still SkyPilot
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import boto3
import requests as _requests
import yaml as _yaml
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/api/v2/serving-configs", tags=["serving-configs"])

S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

AIRFLOW_SERVICE = os.getenv("AIRFLOW_SERVICE", "my-airflow-api-server")
AIRFLOW_NAMESPACE = os.getenv("AIRFLOW_NAMESPACE", "airflow")
AIRFLOW_PORT = os.getenv("AIRFLOW_PORT", "8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
TABULAR_SKY_SERVING_DAG_ID = "tabular_serving_skypilot_pipeline"
TABULAR_SKY_CHAMPION_SYNC_DAG_ID = "tabular_serve_champion_sync"


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


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


# ─── Models ──────────────────────────────────────────────────────────────────


class ServingConfigRequest(BaseModel):
    train_run_id: str
    serving_mode: Literal["ray_only", "kafka"] = "ray_only"
    raw_schema_s3_path: Optional[str] = None   # required when serving_mode == "kafka"
    alias: str = "champion"
    canary: bool = False
    canary_alias: str = "challenger"
    canary_probability: float = Field(0.10, ge=0.0, le=1.0)
    initial_replicas: int = Field(0, ge=0)
    webhook_public_base_url: str = ""
    webhook_path: str = "/infer/webhook"
    webhook_max_timestamp_age_seconds: int = Field(300, ge=1)
    # In-cluster serving is deprecated.
    deployment_target: Literal["skypilot"] = "skypilot"
    min_replicas: int = Field(1, ge=0)
    max_replicas: int = Field(3, ge=1)
    target_qps_per_replica: int = Field(10, ge=1)
    resource_constraints: Optional[dict] = None
    # Optional SkyServe controller override written into ~/.sky/config.yaml.
    # Supported shape:
    # {
    #   "high_availability": true,
    #   "resources": {"infra": "gcp/us-central1", "cpus": "2+", "disk_size": 1024}
    # }
    serve_controller: Optional[dict] = None

    @model_validator(mode="after")
    def validate_kafka_requires_schema(self) -> "ServingConfigRequest":
        if self.serving_mode == "kafka" and not self.raw_schema_s3_path:
            raise ValueError(
                "raw_schema_s3_path is required when serving_mode is 'kafka'"
            )
        return self


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


def _fetch_serving_params(serve_run_id: str, s3, bucket: str) -> dict:
    """Load params_serving.yaml for the given serve_run_id from S3."""
    key = f"runs/serving/{serve_run_id}/params_serving.yaml"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return _yaml.safe_load(obj["Body"].read().decode("utf-8")) or {}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=404,
                detail=f"No params_serving.yaml found for serve_run_id '{serve_run_id}'",
            ) from e
        raise


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def submit_serving_config(request: ServingConfigRequest):
    """Generate params_serving.yaml and save to s3://bucket/runs/serving/{serve_run_id}/."""
    s3 = _s3_client()

    training_params = _fetch_training_params(request.train_run_id, s3, S3_BUCKET)
    lineage = training_params.get("lineage", {})
    dataset = lineage.get("dataset", "")
    preprocess_run_id = lineage.get("preprocess_run_id", "")

    # Auto-extract registry_model_name — try model block first, then kuberay.model block
    registry_model_name = (
        training_params.get("model", {}).get("registry_model_name")
        or training_params.get("kuberay", {}).get("model", {}).get("mlflow_registry_model_name", "")
        or ""
    )

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
        "serving": {
            "serving_mode": request.serving_mode,
            "registry_model_name": registry_model_name,
            "alias": request.alias,
            "canary": request.canary,
            "webhook_public_base_url": request.webhook_public_base_url,
            "webhook_path": request.webhook_path,
            "webhook_max_timestamp_age_seconds": request.webhook_max_timestamp_age_seconds,
            "deployment_target": request.deployment_target,
        },
        "canary": {
            "alias": request.canary_alias,
            "canary_probability": request.canary_probability,
            "initial_replicas": request.initial_replicas,
        },
    }

    # kafka_inference block is only written when serving_mode == "kafka"
    if request.serving_mode == "kafka":
        params["kafka_inference"] = {
            "raw_schema_s3_path": request.raw_schema_s3_path,
        }

    params["skypilot_serving"] = {
        "min_replicas": request.min_replicas,
        "max_replicas": request.max_replicas,
        "target_qps_per_replica": request.target_qps_per_replica,
        "resource_constraints": request.resource_constraints,
        "serve_controller": request.serve_controller,
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
        "serving_mode": request.serving_mode,
        "registry_model_name": registry_model_name,
        "deployment_target": request.deployment_target,
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


@router.post("/{serve_run_id}/deploy", status_code=202)
async def deploy_serving_config(serve_run_id: str):
    """Read params_serving.yaml from S3 and trigger the SkyPilot serving DAG."""
    s3 = _s3_client()
    serving_params = _fetch_serving_params(serve_run_id, s3, S3_BUCKET)

    serving_block = serving_params.get("serving", {})
    lineage = serving_params.get("lineage", {})
    run_meta = serving_params.get("run_metadata", {})
    kafka_block = serving_params.get("kafka_inference", {})
    params_s3_key = f"runs/serving/{serve_run_id}/params_serving.yaml"

    deployment_target = serving_block.get("deployment_target", "skypilot")
    skypilot_block = serving_params.get("skypilot_serving", {}) or {}

    if deployment_target != "skypilot":
        raise HTTPException(
            status_code=410,
            detail=(
                "In-cluster serving deployment is deprecated. "
                "Re-generate params_serving.yaml with deployment_target='skypilot'."
            ),
        )

    dag_conf: dict[str, Any] = {
        "serve_run_id": serve_run_id,
        "train_run_id": lineage.get("train_run_id", ""),
        "dataset": run_meta.get("dataset", ""),
        "serving_mode": serving_block.get("serving_mode", "ray_only"),
        "registry_model_name": serving_block.get("registry_model_name", ""),
        "params_serving_s3_path": f"s3://{S3_BUCKET}/{params_s3_key}",
        "raw_schema_s3_path": kafka_block.get("raw_schema_s3_path"),
        "alias": serving_block.get("alias", "champion"),
        "min_replicas": skypilot_block.get("min_replicas", 1),
        "max_replicas": skypilot_block.get("max_replicas", 3),
        "target_qps_per_replica": skypilot_block.get("target_qps_per_replica", 10),
        "resource_constraints": skypilot_block.get("resource_constraints"),
        "serve_controller": skypilot_block.get("serve_controller"),
    }
    active_dag_id = TABULAR_SKY_SERVING_DAG_ID

    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{active_dag_id}/dagRuns",
            body={
                "logical_date": datetime.now(timezone.utc).isoformat(),
                "conf": dag_conf,
            },
        )
        if status >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Airflow returned {status}: {data}",
            )
        dag_run_id = data.get("dag_run_id", "")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger {active_dag_id} DAG: {exc}",
        ) from exc

    return {"dag_run_id": dag_run_id, "serve_run_id": serve_run_id, "dag_id": active_dag_id}


@router.get("/{serve_run_id}/deploy/{dag_run_id}/status")
async def serving_deploy_status(
    serve_run_id: str,
    dag_run_id: str,
    dag_id: Optional[str] = None,
):
    """Poll Airflow state for a serving deploy DAG run.

    dag_id query param allows polling non-default DAGs (e.g. tabular_serving_skypilot_pipeline).
    """
    target_dag = dag_id or TABULAR_SKY_SERVING_DAG_ID
    try:
        status, data = _airflow_request(
            "GET", f"api/v2/dags/{target_dag}/dagRuns/{dag_run_id}"
        )
        if status >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Airflow returned {status}: {data}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch serving deploy status: {exc}",
        ) from exc

    return {
        "dag_run_id": dag_run_id,
        "serve_run_id": serve_run_id,
        "dag_id": target_dag,
        "state": data.get("state", "unknown"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
    }


# ─── Champion sync endpoint (SkyPilot rolling update) ────────────────────────


class UpdateChampionRequest(BaseModel):
    alias: str = "champion"
    num_nodes: int = Field(1, ge=1)
    resource_constraints: Optional[dict] = None
    serve_controller: Optional[dict] = None


@router.post("/{serve_run_id}/update-champion", status_code=202)
async def update_champion(serve_run_id: str, request: UpdateChampionRequest):
    """Trigger a rolling sky serve update so replicas pick up the current MLflow @champion.

    Reads params_serving.yaml from S3 to resolve registry_model_name, then
    triggers the tabular_serve_champion_sync DAG which calls
    `sky_runner.py update-tabular-serve`.  Zero-downtime — SkyServe keeps old
    replicas serving until new ones are READY.

    Only applicable when deployment_target == "skypilot".
    """
    s3 = _s3_client()
    serving_params = _fetch_serving_params(serve_run_id, s3, S3_BUCKET)

    serving_block = serving_params.get("serving", {})
    skypilot_block = serving_params.get("skypilot_serving", {}) or {}
    deployment_target = serving_block.get("deployment_target", "skypilot")
    if deployment_target != "skypilot":
        raise HTTPException(
            status_code=400,
            detail=(
                f"serve_run_id '{serve_run_id}' has deployment_target='{deployment_target}'. "
                "update-champion is only applicable to SkyPilot deployments."
            ),
        )

    registry_model_name = serving_block.get("registry_model_name", "")
    if not registry_model_name:
        raise HTTPException(
            status_code=400,
            detail="params_serving.yaml is missing 'serving.registry_model_name'.",
        )

    dag_conf: dict[str, Any] = {
        "serve_run_id":         serve_run_id,
        "registry_model_name":  registry_model_name,
        "alias":                request.alias,
        "num_nodes":            request.num_nodes,
    }
    if request.resource_constraints:
        dag_conf["resource_constraints"] = request.resource_constraints

    # Reuse the saved controller config by default so champion sync keeps
    # the same SkyServe controller sizing unless explicitly overridden.
    serve_controller = request.serve_controller
    if serve_controller is None:
        serve_controller = skypilot_block.get("serve_controller")
    if serve_controller:
        dag_conf["serve_controller"] = serve_controller

    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{TABULAR_SKY_CHAMPION_SYNC_DAG_ID}/dagRuns",
            body={
                "logical_date": datetime.now(timezone.utc).isoformat(),
                "conf": dag_conf,
            },
        )
        if status >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Airflow returned {status}: {data}",
            )
        dag_run_id = data.get("dag_run_id", "")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger {TABULAR_SKY_CHAMPION_SYNC_DAG_ID} DAG: {exc}",
        ) from exc

    return {
        "dag_run_id":   dag_run_id,
        "serve_run_id": serve_run_id,
        "dag_id":       TABULAR_SKY_CHAMPION_SYNC_DAG_ID,
        "alias":        request.alias,
    }


# ─── vLLM serving endpoints (Phase 6) ────────────────────────────────────────

VLLM_SERVING_DAG_ID = "vllm_serving_pipeline"


class VllmDeployRequest(BaseModel):
    serve_run_id: str = ""          # auto-generated if empty
    llm_model_id: str
    hf_token: str = ""
    llm_adapter_s3: str = ""        # optional LoRA adapter S3 path
    vllm_port: int = Field(8000, ge=1, le=65535)
    max_model_len: int = Field(4096, ge=256)
    resource_constraints: Optional[dict] = None


class VllmDeployResult(BaseModel):
    serve_run_id: str
    dag_run_id: str
    sky_cluster_name: str


class VllmEndpointResult(BaseModel):
    endpoint_url: str
    model_id: str
    status: str   # "healthy" | "pending" | "not_found"


@router.post("/vllm-deploy", response_model=VllmDeployResult, status_code=202)
async def trigger_vllm_deploy(request: VllmDeployRequest):
    """Trigger the vllm_serving_pipeline Airflow DAG to launch a persistent vLLM cluster."""
    import json as _json

    serve_run_id = request.serve_run_id or (
        "vllm-"
        + secrets.token_hex(3)
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    sky_cluster_name = f"vllm-{serve_run_id[:20]}"

    dag_conf: dict[str, Any] = {
        "serve_run_id": serve_run_id,
        "llm_model_id": request.llm_model_id,
        "hf_token": request.hf_token,
        "llm_adapter_s3": request.llm_adapter_s3,
        "vllm_port": request.vllm_port,
        "max_model_len": request.max_model_len,
    }
    if request.resource_constraints:
        dag_conf["resource_constraints"] = request.resource_constraints

    dag_run_id = (
        f"vllm_serving__{serve_run_id}__"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    try:
        status, data = _airflow_request(
            "POST",
            f"api/v2/dags/{VLLM_SERVING_DAG_ID}/dagRuns",
            body={"dag_run_id": dag_run_id, "conf": dag_conf},
        )
        if status >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Airflow returned {status}: {data}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger vllm_serving_pipeline DAG: {exc}",
        ) from exc

    return VllmDeployResult(
        serve_run_id=serve_run_id,
        dag_run_id=dag_run_id,
        sky_cluster_name=sky_cluster_name,
    )


@router.get("/{serve_run_id}/endpoint", response_model=VllmEndpointResult)
async def get_vllm_endpoint(serve_run_id: str):
    """Fetch the registered vLLM endpoint URL for a serving run from S3."""
    import json as _json

    s3 = _s3_client()
    key = f"runs/serving/{serve_run_id}/endpoint.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = _json.loads(resp["Body"].read())
        return VllmEndpointResult(
            endpoint_url=data.get("endpoint_url", ""),
            model_id=data.get("model_id", ""),
            status="healthy" if data.get("endpoint_url") else "pending",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return VllmEndpointResult(
                endpoint_url="", model_id="", status="not_found"
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
