"""vLLM Serving Pipeline DAG — persistent vLLM cluster via sky.launch().

Unlike training DAGs (managed jobs), this uses sky.launch() to create a
PERSISTENT cluster that keeps running after the DAG completes.

The DAG:
  1. launch_vllm_cluster — provisions the VM with sky.launch() + starts vllm serve
  2. wait_for_endpoint   — polls GET /health until healthy (up to 10 minutes)
  3. register_endpoint   — writes endpoint URL to S3 (for API and UI consumption)

After the DAG succeeds, the cluster continues running.
To tear it down: sky down {cluster_name}  (or use sky.down() in a separate task).

dag_run.conf keys:
    serve_run_id        : str        — unique serving run ID (e.g. vllm-serve-001)
    llm_model_id        : str        — HuggingFace model ID (e.g. Qwen/Qwen2.5-7B)
    hf_token            : str        — HuggingFace token (optional)
    llm_adapter_s3      : str        — s3://bucket/path/adapter/ (optional LoRA adapter)
    vllm_port           : int        — serving port (default 8000)
    max_model_len       : int        — max sequence length (default 4096)
    resource_constraints: dict|None  — optional GPUSelectorService constraints

The endpoint URL is written to:
    s3://{S3_BUCKET}/runs/serving/{serve_run_id}/endpoint.json

Retrieve it via:
    GET /api/v2/serving-configs/{serve_run_id}/endpoint

Requirements in Airflow image:
    pip install "skypilot[runpod,aws,vast]" boto3 requests
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── Constants ────────────────────────────────────────────────────────────────

SKY_YAML_VLLM = Path(
    os.getenv(
        "SKY_VLLM_YAML",
        "/opt/airflow/dags/repo/k3s/sky/vllm-serving.yaml",
    )
)
VLLM_HEALTH_TIMEOUT_SECONDS = int(os.getenv("VLLM_HEALTH_TIMEOUT_SECONDS", "600"))
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_cluster_head_ip(cluster_name: str) -> str | None:
    """Return the head node's external IP for cluster_name, or None on failure."""
    import sky  # type: ignore[import]

    try:
        records = sky.get(sky.status(cluster_names=[cluster_name], refresh=True))
        if not records:
            return None
        handle = records[0].get("handle")
        if handle and hasattr(handle, "external_ips"):
            ips = handle.external_ips()
            if ips:
                return str(ips[0])
    except Exception as exc:
        print(f"[warn] Could not get external IPs for {cluster_name}: {exc}")
    return None


def _load_vllm_task(
    yaml_path: str,
    serve_run_id: str,
    resource_constraints_conf: dict | None,
) -> "sky.Task":  # type: ignore[name-defined]
    """Load a SkyPilot Task for vLLM serving, optionally with dynamic any_of."""
    import sky  # type: ignore[import]
    import yaml as _yaml

    if not resource_constraints_conf:
        return sky.Task.from_yaml(yaml_path)

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    try:
        import dataclasses
        from src.services.gpu_catalog import GPUCatalogService
        from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

        _valid = {f.name for f in dataclasses.fields(ResourceConstraints)}
        merged = {k: v for k, v in resource_constraints_conf.items() if k in _valid}
        # Serving is always on-demand — override prefer_spot
        merged["prefer_spot"] = False
        constraints = ResourceConstraints(**merged)

        offers = GPUCatalogService().query_availability(
            providers=constraints.providers,
            min_vram_gb=constraints.min_vram_gb,
            gpu_types=constraints.gpu_types,
        )
        result = GPUSelectorService().select_providers(constraints, offers)

        if not result.any_of:
            return sky.Task.from_yaml(yaml_path)

        with open(yaml_path) as fh:
            sky_conf = _yaml.safe_load(fh)
        sky_conf.setdefault("resources", {})["any_of"] = result.any_of
        sky_conf["resources"].pop("infra", None)

        tmp = f"/tmp/sky_vllm_{serve_run_id}.yaml"
        with open(tmp, "w") as fh:
            _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)
        return sky.Task.from_yaml(tmp)

    except Exception as exc:
        print(f"[warn] Dynamic resource selection failed ({exc}) — using static YAML")
        return sky.Task.from_yaml(yaml_path)


# ─── Tasks ────────────────────────────────────────────────────────────────────


def _launch_vllm_cluster(**context):
    """Provision the vLLM serving cluster via sky.launch().

    sky.launch() with detach_run=True:
      - Returns after the cluster is provisioned and the run script has started.
      - The cluster stays alive indefinitely (no managed-job auto-cleanup).
    """
    import sky  # type: ignore[import]

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name  # type: ignore[import]

    conf = context["dag_run"].conf or {}
    serve_run_id = conf["serve_run_id"]
    model_id = conf["llm_model_id"]
    hf_token = conf.get("hf_token", "")
    adapter_s3 = conf.get("llm_adapter_s3", "")
    vllm_port = int(conf.get("vllm_port", 8000))
    max_model_len = int(conf.get("max_model_len", 4096))

    # Cluster name: short, RFC-1123 safe
    cluster_name = k8s_name(serve_run_id, "vllm", max_len=32)

    task = _load_vllm_task(
        str(SKY_YAML_VLLM),
        serve_run_id,
        conf.get("resource_constraints"),
    )

    task.update_envs({
        "SERVE_RUN_ID":          serve_run_id,
        "HF_MODEL_ID":           model_id,
        "HF_TOKEN":              hf_token,
        "LLM_ADAPTER_S3":        adapter_s3,
        "VLLM_PORT":             str(vllm_port),
        "MAX_MODEL_LEN":         str(max_model_len),
    })

    # sky.launch() submits the request to the SkyPilot API server (non-blocking).
    # We do NOT call stream_and_get() here because vllm serve runs indefinitely —
    # stream_and_get() would block forever waiting for the run command to exit.
    # Instead, _wait_for_endpoint polls the /health HTTP endpoint.
    sky.launch(
        task,
        cluster_name=cluster_name,
        retry_until_up=True,
    )
    print(f"vLLM cluster launch submitted: {cluster_name}")

    # Try to get the head IP right away (may not be available immediately)
    head_ip = _get_cluster_head_ip(cluster_name)
    print(f"Head IP: {head_ip or '(not yet available)'}")

    context["ti"].xcom_push(key="cluster_name", value=cluster_name)
    context["ti"].xcom_push(key="serve_run_id", value=serve_run_id)
    context["ti"].xcom_push(key="head_ip", value=head_ip)
    context["ti"].xcom_push(key="vllm_port", value=vllm_port)


def _wait_for_endpoint(**context):
    """Poll the vLLM /health endpoint until it returns 200 or times out."""
    import requests  # type: ignore[import]

    ti = context["ti"]
    head_ip = ti.xcom_pull(task_ids="launch_vllm_cluster", key="head_ip")
    cluster_name = ti.xcom_pull(task_ids="launch_vllm_cluster", key="cluster_name")
    vllm_port = ti.xcom_pull(task_ids="launch_vllm_cluster", key="vllm_port") or 8000

    # If head_ip not available yet, try once more from sky.status
    if not head_ip:
        import sky  # type: ignore[import]
        head_ip = _get_cluster_head_ip(cluster_name)

    if not head_ip:
        raise RuntimeError(
            f"Cannot determine head IP for cluster '{cluster_name}'. "
            "Check sky.status() and ensure the cluster has an external IP."
        )

    endpoint = f"http://{head_ip}:{vllm_port}"
    health_url = f"{endpoint}/health"
    print(f"Polling vLLM health: {health_url} (timeout={VLLM_HEALTH_TIMEOUT_SECONDS}s)")

    interval = 15
    elapsed = 0

    while elapsed < VLLM_HEALTH_TIMEOUT_SECONDS:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"vLLM endpoint healthy after {elapsed}s: {endpoint}")
                ti.xcom_push(key="endpoint_url", value=endpoint)
                return
        except Exception as exc:
            pass  # connection refused / not yet ready
        time.sleep(interval)
        elapsed += interval
        print(f"[{elapsed}s] vLLM not ready yet, retrying...")

    raise RuntimeError(
        f"vLLM endpoint '{health_url}' not healthy after {VLLM_HEALTH_TIMEOUT_SECONDS}s."
    )


def _register_endpoint(**context):
    """Write the endpoint URL to S3 so the API and UI can read it."""
    import boto3  # type: ignore[import]

    ti = context["ti"]
    endpoint_url = ti.xcom_pull(task_ids="wait_for_endpoint", key="endpoint_url")
    serve_run_id = ti.xcom_pull(task_ids="launch_vllm_cluster", key="serve_run_id")
    conf = context["dag_run"].conf or {}
    model_id = conf.get("llm_model_id", "")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    bucket = S3_BUCKET
    key = f"runs/serving/{serve_run_id}/endpoint.json"

    payload = json.dumps({
        "endpoint_url": endpoint_url,
        "model_id": model_id,
        "serve_run_id": serve_run_id,
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }).encode()

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
    )
    print(f"Endpoint registered: {endpoint_url} → s3://{bucket}/{key}")


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="vllm_serving_pipeline",
    description=(
        "Provision a persistent vLLM serving cluster via SkyPilot sky.launch(). "
        "On-demand only. Registers endpoint URL to S3 when healthy."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "llm", "vllm", "serving", "skypilot"],
) as dag:

    launch_task = PythonOperator(
        task_id="launch_vllm_cluster",
        python_callable=_launch_vllm_cluster,
    )

    wait_task = PythonOperator(
        task_id="wait_for_endpoint",
        python_callable=_wait_for_endpoint,
        execution_timeout=timedelta(seconds=VLLM_HEALTH_TIMEOUT_SECONDS + 60),
    )

    register_task = PythonOperator(
        task_id="register_endpoint",
        python_callable=_register_endpoint,
    )

    launch_task >> wait_task >> register_task
