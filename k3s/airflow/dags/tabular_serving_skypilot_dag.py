"""Tabular SkyPilot sky serve Pipeline DAG.

Deploys a tabular model (xgboost / sklearn / pytorch) as a managed sky serve
service with auto-scaling replicas via SkyPilot.  Each replica runs Ray Serve
as the HTTP serving framework.

Single-node replica: tabular-serving-single.yaml
Multi-node replica:  tabular-serving-multinode.yaml  (Kimi-K2 pattern adapted)

SkyPilot has a dependency conflict with apache-airflow (protobuf, grpcio).
All sky.* calls are delegated to the takenking9879/sky-runner:0.12.0 pod.

dag_run.conf keys:
    serve_run_id            : str      — unique serving run ID
    registry_model_name     : str      — MLflow registered model name
    alias                   : str      — model alias (default "champion")
    serve_port              : int      — HTTP port for the serve endpoint (default 8000)
    num_nodes               : int      — nodes per replica (1=single, >1=multi-node Ray)
    min_replicas            : int      — minimum replica count (default 1)
    max_replicas            : int      — maximum replica count (default 3)
    target_qps_per_replica  : int      — QPS target for autoscaling (default 10)
    resource_constraints    : dict|None — optional GPUSelectorService constraints

Endpoint written to:
    s3://{S3_BUCKET}/runs/serving/{serve_run_id}/endpoint.json

Retrieve via:
    GET /api/v2/serving-configs/{serve_run_id}/endpoint
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_IMAGE  = os.getenv("SKY_RUNNER_IMAGE", "takenking9879/sky-runner:0.13.0")
_AIRFLOW_NS = os.getenv("AIRFLOW_NAMESPACE", "airflow")
TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS = int(
    os.getenv("TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS", "600")
)
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

# ─── Shared pod helpers ───────────────────────────────────────────────────────

def _aws_env_from() -> list:
    # Injects AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY,
    # VASTAI_API_KEY, MLFLOW_TRACKING_URI from env-secret (.env).
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="env-secret"))]


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

    launch_task = KubernetesPodOperator(
        task_id="launch_tabular_serve",
        name="sky-launch-tabular-serve",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        arguments=["python", "/app/sky_runner.py", "launch-tabular-serve"],
        env_vars={
            "SERVE_RUN_ID":               "{{ dag_run.conf['serve_run_id'] }}",
            "REGISTRY_MODEL_NAME":        "{{ dag_run.conf['registry_model_name'] }}",
            "MODEL_ALIAS":                "{{ dag_run.conf.get('alias', 'champion') }}",
            "SERVE_PORT":                 "{{ dag_run.conf.get('serve_port', 8000) }}",
            "NUM_NODES":                  "{{ dag_run.conf.get('num_nodes', 1) }}",
            "MIN_REPLICAS":               "{{ dag_run.conf.get('min_replicas', 1) }}",
            "MAX_REPLICAS":               "{{ dag_run.conf.get('max_replicas', 3) }}",
            "TARGET_QPS_PER_REPLICA":     "{{ dag_run.conf.get('target_qps_per_replica', 10) }}",
            "RESOURCE_CONSTRAINTS_JSON":  "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
            "S3_BUCKET":                  S3_BUCKET,
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        # sky.serve.up() can take a while to provision replicas
        execution_timeout=timedelta(seconds=TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS + 300),
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "500m", "memory": "1Gi"},
        ),
    )

    wait_task = KubernetesPodOperator(
        task_id="wait_for_endpoint",
        name="sky-wait-tabular-serve",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        arguments=["python", "/app/sky_runner.py", "wait-tabular-serve"],
        env_vars={
            "SERVICE_INFO_JSON": "{{ ti.xcom_pull(task_ids='launch_tabular_serve', key='return_value') | tojson }}",
            "TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS": str(TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS),
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        execution_timeout=timedelta(seconds=TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS + 120),
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "250m", "memory": "512Mi"},
        ),
    )

    register_task = KubernetesPodOperator(
        task_id="register_endpoint",
        name="sky-register-tabular-serve-endpoint",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        arguments=["python", "/app/sky_runner.py", "register-endpoint"],
        env_vars={
            "ENDPOINT_URL":  "{{ ti.xcom_pull(task_ids='wait_for_endpoint', key='return_value') }}",
            "SERVE_RUN_ID":  "{{ dag_run.conf['serve_run_id'] }}",
            "HF_MODEL_ID":   "",   # not applicable for tabular — kept for register_endpoint compat
            "ORCHESTRATION": "tabular_serving_skypilot",
            "S3_BUCKET":     S3_BUCKET,
        },
        env_from=_aws_env_from(),
        get_logs=True,
        is_delete_operator_pod=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "128Mi"},
            limits={"cpu": "200m", "memory": "256Mi"},
        ),
    )

    launch_task >> wait_task >> register_task
