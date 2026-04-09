"""Tabular SkyPilot Champion Sync DAG.

Performs a zero-downtime rolling update of an existing sky serve service so
that replicas pick up the current MLflow @champion model version.

How it works:
  1. fetch_service_info   — reads endpoint.json from S3 to get the service_name
                            that was assigned when the service was originally launched.
  2. update_tabular_serve — runs `sky_runner.py update-tabular-serve` inside a
                            KubernetesPodOperator pod.  SkyPilot calls sky.serve.update(),
                            which triggers a rolling restart of all replicas.  Each new
                            replica runs app.py which loads mlflow.pyfunc using MODEL_ALIAS
                            (default "champion") — picking up the new champion version.

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
from datetime import datetime, timedelta

import boto3
from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from kubernetes.client import models as k8s

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_IMAGE  = os.getenv("SKY_RUNNER_IMAGE", "takenking9879/sky-runner:0.13.0")
_AIRFLOW_NS = os.getenv("AIRFLOW_NAMESPACE", "airflow")
S3_BUCKET   = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

TABULAR_SERVE_UPDATE_TIMEOUT_SECONDS = int(
    os.getenv("TABULAR_SERVE_UPDATE_TIMEOUT_SECONDS", "900")
)


# ─── Shared pod helpers ───────────────────────────────────────────────────────

def _aws_env_from() -> list:
    # Injects AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY,
    # VASTAI_API_KEY, MLFLOW_TRACKING_URI from env-secret (.env).
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="env-secret"))]


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

    update_task = KubernetesPodOperator(
        task_id="update_tabular_serve",
        name="sky-update-tabular-serve",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        arguments=["python", "/app/sky_runner.py", "update-tabular-serve"],
        env_vars={
            # service_name comes from XCom pushed by fetch_service_info
            "SERVICE_NAME":             "{{ ti.xcom_pull(task_ids='fetch_service_info')['service_name'] }}",
            "SERVE_PORT":               "{{ ti.xcom_pull(task_ids='fetch_service_info')['serve_port'] }}",
            "REGISTRY_MODEL_NAME":      "{{ dag_run.conf['registry_model_name'] }}",
            "MODEL_ALIAS":              "{{ dag_run.conf.get('alias', 'champion') }}",
            "NUM_NODES":                "{{ dag_run.conf.get('num_nodes', 1) }}",
            "RESOURCE_CONSTRAINTS_JSON": "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        execution_timeout=timedelta(seconds=TABULAR_SERVE_UPDATE_TIMEOUT_SECONDS),
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "500m", "memory": "1Gi"},
        ),
    )

    fetch_task >> update_task
