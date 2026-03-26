"""vLLM Serving Pipeline DAG — persistent vLLM cluster via KubernetesPodOperator.

SkyPilot has a dependency conflict with apache-airflow (protobuf, grpcio).
All sky.* calls are delegated to the takenking9879/sky-runner:0.12.0 pod.

The DAG provisions a PERSISTENT cluster that keeps running after completion.
sky_runner.py uses sky.launch(detach_run=True) so stream_and_get() returns once
the cluster is up without blocking on the infinite vllm serve process.

dag_run.conf keys:
    serve_run_id        : str        — unique serving run ID
    llm_model_id        : str        — HuggingFace model ID (e.g. Qwen/Qwen2.5-7B)
    hf_token            : str        — HuggingFace token (optional)
    llm_adapter_s3      : str        — s3://bucket/path/adapter/ (optional LoRA)
    vllm_port           : int        — serving port (default 8000)
    max_model_len       : int        — max sequence length (default 4096)
    resource_constraints: dict|None  — optional GPUSelectorService constraints

Tasks:
    1. launch_vllm_cluster — sky-runner pod → sky.launch(detach_run=True); pushes cluster_info (dict)
    2. wait_for_endpoint   — sky-runner pod → polls GET /health; pushes endpoint_url (str)
    3. register_endpoint   — sky-runner pod → writes endpoint.json to S3
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_IMAGE  = os.getenv("SKY_RUNNER_IMAGE", "takenking9879/sky-runner:0.12.0")
_AIRFLOW_NS = os.getenv("AIRFLOW_NAMESPACE", "airflow")
VLLM_HEALTH_TIMEOUT_SECONDS = int(os.getenv("VLLM_HEALTH_TIMEOUT_SECONDS", "600"))
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

# ─── Shared pod helpers ────────────────────────────────────────────────────────

def _aws_env_from() -> list:
    # Injects AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY,
    # VASTAI_API_KEY, MLFLOW_TRACKING_URI from env-secret (.env).
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="env-secret"))]


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="vllm_serving_pipeline",
    description=(
        "Provision a persistent vLLM serving cluster via SkyPilot (KubernetesPodOperator). "
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

    launch_task = KubernetesPodOperator(
        task_id="launch_vllm_cluster",
        name="sky-launch-vllm",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["launch-vllm"],
        env_vars={
            "SERVE_RUN_ID":               "{{ dag_run.conf['serve_run_id'] }}",
            "HF_MODEL_ID":                "{{ dag_run.conf['llm_model_id'] }}",
            "HF_TOKEN":                   "{{ dag_run.conf.get('hf_token', '') }}",
            "LLM_ADAPTER_S3":             "{{ dag_run.conf.get('llm_adapter_s3', '') }}",
            "VLLM_PORT":                  "{{ dag_run.conf.get('vllm_port', 8000) }}",
            "MAX_MODEL_LEN":              "{{ dag_run.conf.get('max_model_len', 4096) }}",
            "RESOURCE_CONSTRAINTS_JSON":  "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "500m", "memory": "1Gi"},
        ),
    )

    wait_task = KubernetesPodOperator(
        task_id="wait_for_endpoint",
        name="sky-wait-vllm",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["wait-vllm"],
        env_vars={
            # cluster_info dict from launch task: {"cluster_name":..., "head_ip":..., "vllm_port":...}
            "CLUSTER_INFO_JSON":              "{{ ti.xcom_pull(task_ids='launch_vllm_cluster', key='return_value') | tojson }}",
            "VLLM_HEALTH_TIMEOUT_SECONDS":    str(VLLM_HEALTH_TIMEOUT_SECONDS),
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        execution_timeout=timedelta(seconds=VLLM_HEALTH_TIMEOUT_SECONDS + 60),
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "250m", "memory": "512Mi"},
        ),
    )

    register_task = KubernetesPodOperator(
        task_id="register_endpoint",
        name="sky-register-endpoint",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["register-endpoint"],
        env_vars={
            # endpoint_url string from wait task
            "ENDPOINT_URL":   "{{ ti.xcom_pull(task_ids='wait_for_endpoint', key='return_value') }}",
            "SERVE_RUN_ID":   "{{ dag_run.conf['serve_run_id'] }}",
            "HF_MODEL_ID":    "{{ dag_run.conf['llm_model_id'] }}",
            "ORCHESTRATION":  "vllm_single_node",
            "S3_BUCKET":      S3_BUCKET,
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
