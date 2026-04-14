"""Tabular SkyPilot sky serve Pipeline DAG.

Runs SkyPilot from the Airflow scheduler via subprocess, matching the
training_pipeline_skypilot pattern. No KubernetesPodOperator tasks are used.

dag_run.conf keys:
    serve_run_id            : str       - unique serving run ID
    registry_model_name     : str       - MLflow registered model name
    alias                   : str       - model alias (default "champion")
    serve_port              : int       - HTTP port for the serve endpoint (default 8000)
    num_nodes               : int       - nodes per replica (1=single, >1=multi-node Ray)
    min_replicas            : int       - minimum replica count (default 1)
    max_replicas            : int       - maximum replica count (default 3)
    target_qps_per_replica  : int       - QPS target for autoscaling (default 10)
    resource_constraints    : dict|None - optional GPUSelectorService constraints

Endpoint written to:
    s3://{S3_BUCKET}/runs/serving/{serve_run_id}/endpoint.json
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_PYTHON = os.getenv("SKYPILOT_VENV_PYTHON", "/opt/skypilot-venv/bin/python")
_SKY_RUNNER_SCRIPT = os.getenv(
    "SKY_RUNNER_SCRIPT", "/opt/airflow/dags/repo/k3s/sky/sky_runner.py"
)
_SKY_YAML_DIR = os.getenv("SKY_YAML_DIR", "/opt/airflow/dags/repo/k3s/sky")

TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS = int(
    os.getenv("TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS", "600")
)
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_sky_runner(command: str, extra_env: dict[str, str]) -> object:
    """Execute sky_runner.py via subprocess and return its XCom payload."""
    xcom_path = "/airflow/xcom/return.json"
    try:
        os.remove(xcom_path)
    except FileNotFoundError:
        pass

    env = os.environ.copy()
    env.update(
        {
            "SKY_YAML_DIR": _SKY_YAML_DIR,
            # Helps optional imports in sky_runner.py (src.services.*)
            "PYTHONPATH": f"/opt/airflow/dags/repo:{env.get('PYTHONPATH', '')}".rstrip(":"),
        }
    )
    env.update({k: str(v) for k, v in extra_env.items() if v is not None})

    cmd = [_SKY_PYTHON, _SKY_RUNNER_SCRIPT, command]
    subprocess.run(cmd, check=True, env=env)

    if not os.path.exists(xcom_path):
        return None

    with open(xcom_path) as fh:
        return json.load(fh)


def _launch_tabular_serve(**context) -> dict:
    conf = context["dag_run"].conf or {}
    result = _run_sky_runner(
        "launch-tabular-serve",
        {
            "SERVE_RUN_ID": conf["serve_run_id"],
            "REGISTRY_MODEL_NAME": conf.get("registry_model_name", ""),
            "MODEL_ALIAS": conf.get("alias", "champion"),
            "SERVE_PORT": conf.get("serve_port", 8000),
            "NUM_NODES": conf.get("num_nodes", 1),
            "MIN_REPLICAS": conf.get("min_replicas", 1),
            "MAX_REPLICAS": conf.get("max_replicas", 3),
            "TARGET_QPS_PER_REPLICA": conf.get("target_qps_per_replica", 10),
            "RESOURCE_CONSTRAINTS_JSON": json.dumps(conf.get("resource_constraints") or {}),
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
