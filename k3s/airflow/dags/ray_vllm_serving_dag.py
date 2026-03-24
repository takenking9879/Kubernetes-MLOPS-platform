"""Ray+vLLM Multi-Node Serving Pipeline DAG (Kimi-K2 pattern).

Uses sky.launch() (persistent cluster) to deploy vLLM across multiple nodes
with tensor + pipeline parallelism.

Unlike the single-node vllm_serving_pipeline, this DAG:
  - Provisions num_nodes machines (head + workers)
  - Head node starts Ray + vllm serve with --pipeline-parallel-size=num_nodes
  - Workers join the Ray cluster and idle (Ray routes work internally)
  - Endpoint is accessible on the head node's external IP

Use this for models that don't fit on a single node.
For single-node, use vllm_serving_pipeline instead.

dag_run.conf keys:
    serve_run_id          : str   — unique serving run ID
    llm_model_id          : str   — HuggingFace model ID
    hf_token              : str   — HuggingFace token (optional)
    llm_adapter_s3        : str   — s3://bucket/path/adapter/ (optional LoRA)
    vllm_port             : int   — serving port (default 8081)
    max_model_len         : int   — max context length (default 32768)
    tensor_parallel_size  : int   — GPUs per node (default 8)
    pipeline_parallel_size: int   — number of nodes (default 2)
    resource_constraints  : dict|None — optional GPUSelectorService constraints

Endpoint written to:
    s3://{S3_BUCKET}/runs/serving/{serve_run_id}/endpoint.json

Retrieve via:
    GET /api/v2/serving-configs/{serve_run_id}/endpoint
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

SKY_YAML_VLLM_MULTI = Path(
    os.getenv(
        "SKY_VLLM_MULTI_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-vllm-multinode-serving.yaml",
    )
)
VLLM_HEALTH_TIMEOUT_SECONDS = int(os.getenv("VLLM_MULTI_HEALTH_TIMEOUT_SECONDS", "900"))
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

_DAGS_DIR = Path(__file__).parent.parent         # k3s/airflow/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_cluster_head_ip(cluster_name: str) -> str | None:
    """Return the head node's external IP, or None on failure."""
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


def _load_task(
    yaml_path: str,
    serve_run_id: str,
    conf: dict,
) -> "sky.Task":  # type: ignore[name-defined]
    """Load SkyPilot Task, optionally injecting dynamic any_of."""
    import sky  # type: ignore[import]
    import yaml as _yaml

    resource_constraints_conf = conf.get("resource_constraints")
    tensor_parallel = int(conf.get("tensor_parallel_size", 8))
    pipeline_parallel = int(conf.get("pipeline_parallel_size", 2))

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    sky_conf: dict | None = None

    if resource_constraints_conf:
        try:
            import dataclasses
            from src.services.gpu_catalog import GPUCatalogService
            from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

            _valid = {f.name for f in dataclasses.fields(ResourceConstraints)}
            merged = {k: v for k, v in resource_constraints_conf.items() if k in _valid}
            merged["prefer_spot"] = False   # serving is always on-demand
            constraints = ResourceConstraints(**merged)

            offers = GPUCatalogService().query_availability(
                providers=constraints.providers,
                min_vram_gb=constraints.min_vram_gb,
                gpu_types=constraints.gpu_types,
            )
            result = GPUSelectorService().select_providers(constraints, offers)

            if result.any_of:
                with open(yaml_path) as fh:
                    sky_conf = _yaml.safe_load(fh)
                sky_conf.setdefault("resources", {})["any_of"] = result.any_of
                sky_conf["resources"].pop("infra", None)
        except Exception as exc:
            print(f"[warn] Dynamic resource selection failed ({exc}) — using static YAML")

    if sky_conf is None:
        with open(yaml_path) as fh:
            sky_conf = _yaml.safe_load(fh)

    # Set num_nodes from pipeline_parallel_size
    sky_conf["num_nodes"] = pipeline_parallel

    tmp = f"/tmp/sky_ray_vllm_{serve_run_id}.yaml"
    with open(tmp, "w") as fh:
        _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)
    return sky.Task.from_yaml(tmp)


# ─── Tasks ────────────────────────────────────────────────────────────────────


def _launch_cluster(**context):
    """Provision multi-node vLLM cluster via sky.launch()."""
    import sky  # type: ignore[import]

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name  # type: ignore[import]

    conf = context["dag_run"].conf or {}
    serve_run_id = conf["serve_run_id"]
    model_id = conf["llm_model_id"]
    hf_token = conf.get("hf_token", "")
    adapter_s3 = conf.get("llm_adapter_s3", "")
    vllm_port = int(conf.get("vllm_port", 8081))
    max_model_len = int(conf.get("max_model_len", 32768))
    tensor_parallel = int(conf.get("tensor_parallel_size", 8))
    pipeline_parallel = int(conf.get("pipeline_parallel_size", 2))

    cluster_name = k8s_name(serve_run_id, "rvllm", max_len=32)
    task = _load_task(str(SKY_YAML_VLLM_MULTI), serve_run_id, conf)

    task.update_envs({
        "SERVE_RUN_ID":            serve_run_id,
        "HF_MODEL_ID":             model_id,
        "HF_TOKEN":                hf_token,
        "LLM_ADAPTER_S3":          adapter_s3,
        "VLLM_PORT":               str(vllm_port),
        "MAX_MODEL_LEN":           str(max_model_len),
        "TENSOR_PARALLEL_SIZE":    str(tensor_parallel),
        "PIPELINE_PARALLEL_SIZE":  str(pipeline_parallel),
    })

    # Non-blocking submit — see vllm_serving_dag.py comment on why no stream_and_get.
    sky.launch(
        task,
        cluster_name=cluster_name,
        retry_until_up=True,
    )
    print(f"Ray+vLLM cluster launch submitted: {cluster_name} ({pipeline_parallel} nodes)")

    head_ip = _get_cluster_head_ip(cluster_name)
    context["ti"].xcom_push(key="cluster_name", value=cluster_name)
    context["ti"].xcom_push(key="serve_run_id", value=serve_run_id)
    context["ti"].xcom_push(key="head_ip", value=head_ip)
    context["ti"].xcom_push(key="vllm_port", value=vllm_port)


def _wait_for_endpoint(**context):
    """Poll the vLLM /health endpoint until healthy or timeout."""
    import requests  # type: ignore[import]
    import sky  # type: ignore[import]

    ti = context["ti"]
    head_ip = ti.xcom_pull(task_ids="launch_cluster", key="head_ip")
    cluster_name = ti.xcom_pull(task_ids="launch_cluster", key="cluster_name")
    vllm_port = ti.xcom_pull(task_ids="launch_cluster", key="vllm_port") or 8081

    if not head_ip:
        head_ip = _get_cluster_head_ip(cluster_name)

    if not head_ip:
        raise RuntimeError(
            f"Cannot determine head IP for cluster '{cluster_name}'. "
            "Check sky.status() and ensure the cluster has an external IP."
        )

    endpoint = f"http://{head_ip}:{vllm_port}"
    health_url = f"{endpoint}/health"
    print(
        f"Polling Ray+vLLM health: {health_url} "
        f"(timeout={VLLM_HEALTH_TIMEOUT_SECONDS}s)"
    )

    interval = 20   # multi-node startup takes longer
    elapsed = 0

    while elapsed < VLLM_HEALTH_TIMEOUT_SECONDS:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"Ray+vLLM endpoint healthy after {elapsed}s: {endpoint}")
                ti.xcom_push(key="endpoint_url", value=endpoint)
                return
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
        print(f"[{elapsed}s] Ray+vLLM not ready yet, retrying...")

    raise RuntimeError(
        f"Ray+vLLM endpoint '{health_url}' not healthy after "
        f"{VLLM_HEALTH_TIMEOUT_SECONDS}s."
    )


def _register_endpoint(**context):
    """Write the endpoint URL to S3 for the API and UI."""
    import boto3  # type: ignore[import]

    ti = context["ti"]
    endpoint_url = ti.xcom_pull(task_ids="wait_for_endpoint", key="endpoint_url")
    serve_run_id = ti.xcom_pull(task_ids="launch_cluster", key="serve_run_id")
    cluster_name = ti.xcom_pull(task_ids="launch_cluster", key="cluster_name")
    conf = context["dag_run"].conf or {}
    model_id = conf.get("llm_model_id", "")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    key = f"runs/serving/{serve_run_id}/endpoint.json"
    payload = json.dumps({
        "endpoint_url": endpoint_url,
        "model_id": model_id,
        "serve_run_id": serve_run_id,
        "cluster_name": cluster_name,
        "orchestration": "ray_vllm_multinode",
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }).encode()

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=payload,
        ContentType="application/json",
    )
    print(f"Endpoint registered: {endpoint_url} → s3://{S3_BUCKET}/{key}")


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="ray_vllm_serving_pipeline",
    description=(
        "Multi-node Ray+vLLM serving cluster (Kimi-K2 pattern). "
        "On-demand only. Tensor + pipeline parallelism. "
        "Registers endpoint URL to S3 when healthy."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "llm", "vllm", "ray", "multinode", "serving", "skypilot"],
) as dag:

    launch_task = PythonOperator(
        task_id="launch_cluster",
        python_callable=_launch_cluster,
    )

    wait_task = PythonOperator(
        task_id="wait_for_endpoint",
        python_callable=_wait_for_endpoint,
        execution_timeout=timedelta(seconds=VLLM_HEALTH_TIMEOUT_SECONDS + 120),
    )

    register_task = PythonOperator(
        task_id="register_endpoint",
        python_callable=_register_endpoint,
    )

    launch_task >> wait_task >> register_task
