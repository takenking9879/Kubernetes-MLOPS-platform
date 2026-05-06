"""Tabular SkyPilot sky serve Pipeline DAG.

Runs SkyPilot from the Airflow scheduler via subprocess, matching the
training_pipeline_skypilot pattern. No KubernetesPodOperator tasks are used.

dag_run.conf keys:
    serve_run_id            : str       - unique serving run ID
    registry_model_name     : str       - MLflow registered model name
    mlflow_tracking_uri     : str       - optional MLflow URI override for remote serving replicas
    alias                   : str       - model alias (default "champion")
    serve_port              : int       - HTTP port for the serve endpoint (default 8000)
    num_nodes               : int       - nodes per replica (1=single, >1=multi-node Ray)
    min_replicas            : int       - minimum replica count (default 1)
    max_replicas            : int       - maximum replica count (default 3)
    target_qps_per_replica  : int       - QPS target for autoscaling (default 10)
    resource_constraints    : dict|None - optional GPUSelectorService constraints
    serve_controller        : dict|None - optional SkyServe controller config override

Logs during wait_for_endpoint are streamed live from SkyServe
(controller/load balancer/replicas) by sky_runner.py.

Endpoint written to:
    s3://{S3_BUCKET}/runs/serving/{serve_run_id}/endpoint.json
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_PYTHON = os.getenv("SKYPILOT_VENV_PYTHON", "/opt/skypilot-venv/bin/python")
_AIRFLOW_PYTHON = os.getenv("AIRFLOW_PYTHON", sys.executable)
_SKY_RUNNER_SCRIPT = os.getenv(
    "SKY_RUNNER_SCRIPT", "/opt/airflow/dags/repo/k3s/sky/sky_runner.py"
)
_SKY_YAML_DIR = os.getenv("SKY_YAML_DIR", "/opt/airflow/dags/repo/k3s/sky")

TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS = int(
    os.getenv("TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS", "600")
)
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_sky_runner(
    command: str,
    extra_env: dict[str, str],
    python_bin: str | None = None,
) -> object:
    """Execute sky_runner.py via subprocess and return its XCom payload."""
    fd, xcom_path = tempfile.mkstemp(prefix=f"sky_runner_{command}_", suffix=".json")
    os.close(fd)
    try:
        os.remove(xcom_path)
    except FileNotFoundError:
        pass

    env = os.environ.copy()
    env.update(
        {
            "SKY_YAML_DIR": _SKY_YAML_DIR,
            "SKY_RUNNER_XCOM_PATH": xcom_path,
            # Force unbuffered stdout/stderr so live logs appear in Airflow
            # while wait-tabular-serve is running.
            "PYTHONUNBUFFERED": "1",
            # Helps optional imports in sky_runner.py (src.services.*)
            "PYTHONPATH": f"/opt/airflow/dags/repo:{env.get('PYTHONPATH', '')}".rstrip(":"),
        }
    )
    env.update({k: str(v) for k, v in extra_env.items() if v is not None})

    runner_python = python_bin or _SKY_PYTHON
    cmd = [runner_python, "-u", _SKY_RUNNER_SCRIPT, command]
    subprocess.run(cmd, check=True, env=env)

    if not os.path.exists(xcom_path):
        return None

    with open(xcom_path) as fh:
        return json.load(fh)


def _parse_k8s_cpu_count(raw_cpu: str | int | float | None) -> int:
    """Normalize Kubernetes CPU quantity strings to whole CPU cores."""
    if raw_cpu is None:
        return 0
    raw = str(raw_cpu).strip()
    if not raw:
        return 0
    if raw.endswith("m"):
        milli = raw[:-1].strip()
        if not milli:
            return 0
        return max(1, int(float(milli) / 1000.0))
    return max(1, int(float(raw)))


def _detect_k3s_gpu_for_serve(rc: dict, n_gpus: int = 1) -> tuple[str, int]:
    """Detect K8s GPU accelerator label and local CPU budget."""
    from kubernetes import client as _k8s_client, config as _k8s_config
    k8s_acc = ""
    k8s_cpu_budget = 0
    try:
        try:
            _k8s_config.load_incluster_config()
        except Exception:
            _k8s_config.load_kube_config()
        v1 = _k8s_client.CoreV1Api()
        for node in v1.list_node().items:
            labels = node.metadata.labels or {}
            capacity = node.status.capacity or {}
            allocatable = node.status.allocatable or {}
            if int(capacity.get("nvidia.com/gpu", 0) or 0) == 0:
                continue
            acc = labels.get("skypilot.co/accelerator", "")
            if acc:
                k8s_acc = f"{acc}:{n_gpus}"
                node_cpus = _parse_k8s_cpu_count(
                    allocatable.get("cpu") or capacity.get("cpu")
                )
                k8s_cpu_budget = max(1, node_cpus // 3) if node_cpus > 0 else 1
                break
    except Exception as exc:
        raise RuntimeError(f"[tabular-serve] K8s node query failed: {exc}") from exc
    if not k8s_acc:
        raise RuntimeError(
            "[tabular-serve] localGPU selected but no K8s node with nvidia.com/gpu "
            "and skypilot.co/accelerator label found."
        )
    return k8s_acc, k8s_cpu_budget


def _launch_tabular_serve(**context) -> dict:
    conf = context["dag_run"].conf or {}
    mlflow_tracking_uri = (
        (conf.get("mlflow_tracking_uri") or "").strip()
        or (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    )

    rc = conf.get("resource_constraints") or {}
    providers = rc.get("providers") or []
    provider = str(providers[0]).lower() if providers else "runpod"

    extra: dict = {}
    if provider in ("localgpu", "k8s"):
        n_gpus = max(1, int(rc.get("num_gpus_per_node") or 1))
        k8s_acc, k8s_cpu_budget = _detect_k3s_gpu_for_serve(rc, n_gpus)
        extra["K8S_GPU_ACCELERATOR"] = k8s_acc
        extra["K8S_CPU_LIMIT"] = str(k8s_cpu_budget)
        print(f"[tabular-serve] K3S GPU detected: {k8s_acc} (cpus={k8s_cpu_budget})")

    result = _run_sky_runner(
        "launch-tabular-serve",
        {
            "SERVE_RUN_ID": conf["serve_run_id"],
            "REGISTRY_MODEL_NAME": conf.get("registry_model_name", ""),
            "MLFLOW_TRACKING_URI": mlflow_tracking_uri,
            "MODEL_ALIAS": conf.get("alias", "champion"),
            "SERVE_PORT": conf.get("serve_port", 8000),
            "NUM_NODES": conf.get("num_nodes", 1),
            "MIN_REPLICAS": conf.get("min_replicas", 1),
            "MAX_REPLICAS": conf.get("max_replicas", 3),
            "TARGET_QPS_PER_REPLICA": conf.get("target_qps_per_replica", 10),
            "RESOURCE_CONSTRAINTS_JSON": json.dumps(conf.get("resource_constraints") or {}),
            "SKYSERVE_CONTROLLER_CONFIG_JSON": json.dumps(conf.get("serve_controller") or {}),
            **extra,
        },
    )
    if not isinstance(result, dict):
        raise RuntimeError("launch-tabular-serve did not return service metadata")
    return result


def _wait_for_endpoint(**context) -> str:
    ti = context["ti"]
    service_info = ti.xcom_pull(task_ids="launch_tabular_serve") or {}
    result = _run_sky_runner(
        "wait-tabular-serve",
        {
            "SERVICE_INFO_JSON": json.dumps(service_info),
            "TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS": TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS,
        },
    )
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("wait-tabular-serve did not return endpoint URL")
    return result.strip()


def _register_endpoint(**context) -> None:
    conf = context["dag_run"].conf or {}
    ti = context["ti"]
    endpoint_url = (ti.xcom_pull(task_ids="wait_for_endpoint") or "").strip()
    service_info = ti.xcom_pull(task_ids="launch_tabular_serve") or {}

    if not endpoint_url:
        raise RuntimeError("Missing endpoint URL from wait_for_endpoint")

    _run_sky_runner(
        "register-endpoint",
        {
            "ENDPOINT_URL": endpoint_url,
            "SERVE_RUN_ID": conf["serve_run_id"],
            "HF_MODEL_ID": "",
            "SERVICE_NAME": service_info.get("service_name", ""),
            "ORCHESTRATION": "tabular_serving_skypilot",
            "S3_BUCKET": S3_BUCKET,
        },
        python_bin=_AIRFLOW_PYTHON,
    )


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="tabular_serving_skypilot_pipeline",
    description=(
        "Tabular model sky serve pipeline — SkyPilot managed replicas with Ray Serve. "
        "Single-node or multi-node (Kimi-K2 pattern). "
        "Registers endpoint URL to S3 when healthy."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "tabular", "ray", "serving", "skypilot", "sky-serve"],
) as dag:

    launch_task = PythonOperator(
        task_id="launch_tabular_serve",
        python_callable=_launch_tabular_serve,
        execution_timeout=timedelta(seconds=TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS + 300),
    )

    wait_task = PythonOperator(
        task_id="wait_for_endpoint",
        python_callable=_wait_for_endpoint,
        execution_timeout=timedelta(seconds=TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS + 120),
    )

    register_task = PythonOperator(
        task_id="register_endpoint",
        python_callable=_register_endpoint,
        execution_timeout=timedelta(seconds=120),
    )

    launch_task >> wait_task >> register_task
