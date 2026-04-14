"""Tabular SkyPilot Champion Sync DAG.

Performs a zero-downtime rolling update of an existing sky serve service so
that replicas pick up the current MLflow @champion model version.

How it works:
  1. fetch_service_info — reads endpoint.json from S3 to get the service_name
      assigned when the service was originally launched.
  2. update_tabular_serve — runs `sky_runner.py update-tabular-serve` via
      subprocess from the scheduler process. SkyPilot calls sky.serve.update(),
      which triggers a rolling restart of replicas. New replicas load
      mlflow.pyfunc using MODEL_ALIAS (default "champion").

Trigger this DAG after promoting a new model version to @champion in MLflow so that
the SkyPilot serving deployment stays in sync with the registry.

dag_run.conf keys:
    serve_run_id          : str      — used to look up service_name in S3 endpoint.json
    registry_model_name   : str      — MLflow registered model name
    alias                 : str      — model alias (default "champion")
    num_nodes             : int      — nodes per SkyServe replica (default 1, must match launch)
    resource_constraints  : dict|None — optional GPUSelectorService constraints

Endpoint.json location:
    s3://{S3_BUCKET}/runs/serving/{serve_run_id}/endpoint.json
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta

import boto3
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_PYTHON = os.getenv("SKYPILOT_VENV_PYTHON", "/opt/skypilot-venv/bin/python")
_SKY_RUNNER_SCRIPT = os.getenv(
    "SKY_RUNNER_SCRIPT", "/opt/airflow/dags/repo/k3s/sky/sky_runner.py"
)
_SKY_YAML_DIR = os.getenv("SKY_YAML_DIR", "/opt/airflow/dags/repo/k3s/sky")
S3_BUCKET   = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

TABULAR_SERVE_UPDATE_TIMEOUT_SECONDS = int(
    os.getenv("TABULAR_SERVE_UPDATE_TIMEOUT_SECONDS", "900")
)


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


# ─── Task callables ───────────────────────────────────────────────────────────

def _fetch_service_info(**context) -> dict:
    """Read endpoint.json from S3 and return service info dict.

    Pushes {"service_name": "...", "serve_port": ...} to XCom so the
    update_tabular_serve task knows which sky serve service to update.
    """
    conf        = context["dag_run"].conf or {}
    serve_run_id = conf.get("serve_run_id", "")
    if not serve_run_id:
        raise ValueError("dag_run.conf must contain 'serve_run_id'")

    bucket = os.getenv("S3_BUCKET", S3_BUCKET)
    key    = f"runs/serving/{serve_run_id}/endpoint.json"

    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(obj["Body"].read())

    service_name = payload.get("service_name", "")
    serve_port   = payload.get("serve_port", 8000)

    if not service_name:
        raise RuntimeError(
            f"endpoint.json at s3://{bucket}/{key} does not contain 'service_name'. "
            "Make sure the service was launched via tabular_serving_skypilot_pipeline."
        )

    print(f"[fetch_service_info] service_name={service_name} serve_port={serve_port}")
    return {"service_name": service_name, "serve_port": serve_port}


def _update_tabular_serve(**context) -> dict:
    conf = context["dag_run"].conf or {}
    service_info = context["ti"].xcom_pull(task_ids="fetch_service_info") or {}

    service_name = service_info.get("service_name", "")
    if not service_name:
        raise RuntimeError("Missing service_name from fetch_service_info")

    result = _run_sky_runner(
        "update-tabular-serve",
        {
            "SERVICE_NAME": service_name,
            "SERVE_PORT": service_info.get("serve_port", 8000),
            "REGISTRY_MODEL_NAME": conf["registry_model_name"],
            "MODEL_ALIAS": conf.get("alias", "champion"),
            "NUM_NODES": conf.get("num_nodes", 1),
            "RESOURCE_CONSTRAINTS_JSON": json.dumps(conf.get("resource_constraints") or {}),
        },
    )
    if not isinstance(result, dict):
        return {"service_name": service_name, "status": "updated"}
    return result


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="tabular_serve_champion_sync",
    description=(
        "Zero-downtime rolling update of a SkyPilot sky serve service to pick up "
        "the current MLflow @champion model.  Trigger after promoting a new champion."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=3),
    },
    tags=["mlops", "tabular", "ray", "serving", "skypilot", "sky-serve", "champion"],
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_service_info",
        python_callable=_fetch_service_info,
    )

    update_task = PythonOperator(
        task_id="update_tabular_serve",
        python_callable=_update_tabular_serve,
        execution_timeout=timedelta(seconds=TABULAR_SERVE_UPDATE_TIMEOUT_SECONDS),
    )

    fetch_task >> update_task
