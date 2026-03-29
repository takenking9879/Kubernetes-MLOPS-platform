"""Training Pipeline DAG — SkyPilot via KubernetesPodOperator.

SkyPilot has a dependency conflict with apache-airflow (protobuf, grpcio).
All sky.* calls are delegated to the takenking9879/sky-runner:0.12.0 pod
which runs in the airflow namespace via KubernetesPodOperator.

dag_run.conf (same as before):
    train_run_id            : str        — unique training run ID
    preprocess_run_id       : str        — referenced preprocessing run ID
    dataset                 : str        — canonical dataset name
    processed_table         : str        — Iceberg processed table to train on
    dsl_s3_path             : str        — S3 URI to DSL YAML
    train_params_s3_path    : str        — S3 URI of params_training.yaml
    model_type              : str        — "xgboost" → CPU  |  "pytorch/ssm/bae" → GPU
    num_nodes               : int        — number of nodes (default 1)
    resource_constraints    : dict|None  — optional GPUSelectorService constraints

Tasks:
    1. submit_sky_job  — launches sky-runner pod → sky.jobs.launch(); pushes job_name to XCom
    2. poll_sky_job    — launches sky-runner pod → polls sky.jobs.queue() until terminal state
    3. cleanup_skypilot_controller_on_failure — best-effort controller teardown on DAG failure

Prerequisites:
    - Secret "sky-config"  in airflow namespace (from ~/.sky/config.yaml)
    - Secret "env-secret"  in airflow namespace (AWS creds, MLFLOW_TRACKING_URI)
    - RBAC: my-airflow-scheduler can create/delete pods in airflow namespace
    - Image takenking9879/sky-runner:0.12.0 pushed to DockerHub
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_IMAGE   = os.getenv("SKY_RUNNER_IMAGE", "takenking9879/sky-runner:0.12.0")
_AIRFLOW_NS  = os.getenv("AIRFLOW_NAMESPACE", "airflow")
SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_TIMEOUT_SECONDS", "7200"))
_SKY_IMAGE_PULL_POLICY = os.getenv("SKY_RUNNER_IMAGE_PULL_POLICY", "IfNotPresent")

# SkyPilot API server can transiently use >1Gi memory during startup.
_SKY_SUBMIT_REQUESTS = {
    "cpu": os.getenv("SKY_SUBMIT_CPU_REQUEST", "500m"),
    "memory": os.getenv("SKY_SUBMIT_MEMORY_REQUEST", "2Gi"),
}
_SKY_SUBMIT_LIMITS = {
    "cpu": os.getenv("SKY_SUBMIT_CPU_LIMIT", "750m"),
    "memory": os.getenv("SKY_SUBMIT_MEMORY_LIMIT", "3Gi"),
}

_SKY_POLL_REQUESTS = {
    "cpu": os.getenv("SKY_POLL_CPU_REQUEST", "500m"),
    "memory": os.getenv("SKY_POLL_MEMORY_REQUEST", "2Gi"),
}
_SKY_POLL_LIMITS = {
    "cpu": os.getenv("SKY_POLL_CPU_LIMIT", "750m"),
    "memory": os.getenv("SKY_POLL_MEMORY_LIMIT", "3Gi"),
}

# ─── Shared pod configuration helpers ─────────────────────────────────────────

def _aws_env_from() -> list:
    # Injects AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY,
    # VASTAI_API_KEY, MLFLOW_TRACKING_URI — all from env-secret (.env file).
    # The entrypoint script writes ~/.runpod/config.toml and
    # ~/.config/vastai/vast_api_key from RUNPOD_API_KEY / VASTAI_API_KEY.
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="env-secret"))]


# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline_skypilot",
    description=(
        "SkyPilot managed-jobs training pipeline (KubernetesPodOperator). "
        "Routes pytorch/ssm/bae → GPU spot-first, xgboost → CPU on RunPod."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "training", "skypilot"],
) as dag:

    submit_sky = KubernetesPodOperator(
        task_id="submit_sky_job",
        name="sky-submit-training",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy=_SKY_IMAGE_PULL_POLICY,
        arguments=["python", "/app/sky_runner.py", "submit-training"],
        env_vars={
            "TRAIN_RUN_ID":               "{{ dag_run.conf['train_run_id'] }}",
            "PREPROCESS_RUN_ID":          "{{ dag_run.conf['preprocess_run_id'] }}",
            "PROCESSED_TABLE":            "{{ dag_run.conf.get('processed_table', '') }}",
            "MODEL_TYPE":                 "{{ dag_run.conf.get('model_type', 'xgboost') }}",
            "PARAMS_S3_PATH":             "{{ dag_run.conf['train_params_s3_path'] }}",
            "NUM_NODES":                  "{{ dag_run.conf.get('num_nodes', 1) }}",
            "RESOURCE_CONSTRAINTS_JSON":  "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
            "SKY_TIMEOUT_SECONDS":        str(SKY_TIMEOUT_SECONDS),
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        container_resources=k8s.V1ResourceRequirements(
            requests=_SKY_SUBMIT_REQUESTS,
            limits=_SKY_SUBMIT_LIMITS,
        ),
    )

    poll_sky = KubernetesPodOperator(
        task_id="poll_sky_job",
        name="sky-poll-training",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy=_SKY_IMAGE_PULL_POLICY,
        arguments=["python", "/app/sky_runner.py", "poll-training"],
        env_vars={
            # return_value from submit task is the job_name string
            "SKY_JOB_NAME":        "{{ ti.xcom_pull(task_ids='submit_sky_job', key='return_value') }}",
            "SKY_TIMEOUT_SECONDS": str(SKY_TIMEOUT_SECONDS),
        },
        env_from=_aws_env_from(),
        get_logs=True,
        is_delete_operator_pod=True,
        execution_timeout=timedelta(seconds=SKY_TIMEOUT_SECONDS + 300),
        container_resources=k8s.V1ResourceRequirements(
            requests=_SKY_POLL_REQUESTS,
            limits=_SKY_POLL_LIMITS,
        ),
    )

    cleanup_on_failure = KubernetesPodOperator(
        task_id="cleanup_skypilot_controller_on_failure",
        name="sky-cleanup-training-controller",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy=_SKY_IMAGE_PULL_POLICY,
        arguments=["python", "/app/sky_runner.py", "cleanup-failed-training-controller"],
        env_vars={
            "SKY_JOB_NAME": "{{ ti.xcom_pull(task_ids='submit_sky_job', key='return_value') or '' }}",
        },
        env_from=_aws_env_from(),
        get_logs=True,
        is_delete_operator_pod=True,
        trigger_rule=TriggerRule.ONE_FAILED,
        container_resources=k8s.V1ResourceRequirements(
            requests=_SKY_POLL_REQUESTS,
            limits=_SKY_POLL_LIMITS,
        ),
    )

    submit_sky >> poll_sky
    [submit_sky, poll_sky] >> cleanup_on_failure
