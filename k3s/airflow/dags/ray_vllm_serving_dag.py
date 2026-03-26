"""Ray+vLLM Multi-Node Serving Pipeline DAG (Kimi-K2 pattern).

SkyPilot has a dependency conflict with apache-airflow (protobuf, grpcio).
All sky.* calls are delegated to the takenking9879/sky-runner:0.12.0 pod.

Provisions a PERSISTENT multi-node cluster via sky.launch(detach_run=True).
Head node starts Ray + vllm serve with --pipeline-parallel-size=num_nodes.
Workers join the Ray cluster and idle; Ray routes work internally.

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

import os
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ─── Constants ────────────────────────────────────────────────────────────────

_SKY_IMAGE  = os.getenv("SKY_RUNNER_IMAGE", "takenking9879/sky-runner:0.12.0")
_AIRFLOW_NS = os.getenv("AIRFLOW_NAMESPACE", "airflow")
VLLM_HEALTH_TIMEOUT_SECONDS = int(os.getenv("VLLM_MULTI_HEALTH_TIMEOUT_SECONDS", "900"))
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")

# ─── Shared pod helpers ────────────────────────────────────────────────────────

def _aws_env_from() -> list:
    # Injects AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY,
    # VASTAI_API_KEY, MLFLOW_TRACKING_URI from env-secret (.env).
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="env-secret"))]


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="ray_vllm_serving_pipeline",
    description=(
        "Multi-node Ray+vLLM serving cluster — Kimi-K2 pattern (KubernetesPodOperator). "
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

    launch_task = KubernetesPodOperator(
        task_id="launch_cluster",
        name="sky-launch-ray-vllm",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["launch-ray-vllm"],
        env_vars={
            "SERVE_RUN_ID":               "{{ dag_run.conf['serve_run_id'] }}",
            "HF_MODEL_ID":                "{{ dag_run.conf['llm_model_id'] }}",
            "HF_TOKEN":                   "{{ dag_run.conf.get('hf_token', '') }}",
            "LLM_ADAPTER_S3":             "{{ dag_run.conf.get('llm_adapter_s3', '') }}",
            "VLLM_PORT":                  "{{ dag_run.conf.get('vllm_port', 8081) }}",
            "MAX_MODEL_LEN":              "{{ dag_run.conf.get('max_model_len', 32768) }}",
            "TENSOR_PARALLEL_SIZE":       "{{ dag_run.conf.get('tensor_parallel_size', 8) }}",
            "PIPELINE_PARALLEL_SIZE":     "{{ dag_run.conf.get('pipeline_parallel_size', 2) }}",
            "RESOURCE_CONSTRAINTS_JSON":  "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "500m", "memory": "1Gi"},
        ),
    )

    wait_task = KubernetesPodOperator(
        task_id="wait_for_endpoint",
        name="sky-wait-ray-vllm",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["wait-ray-vllm"],
        env_vars={
            "CLUSTER_INFO_JSON":           "{{ ti.xcom_pull(task_ids='launch_cluster', key='return_value') | tojson }}",
            "VLLM_HEALTH_TIMEOUT_SECONDS": str(VLLM_HEALTH_TIMEOUT_SECONDS),
        },
        env_from=_aws_env_from(),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        execution_timeout=timedelta(seconds=VLLM_HEALTH_TIMEOUT_SECONDS + 120),
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "250m", "memory": "512Mi"},
        ),
    )

    register_task = KubernetesPodOperator(
        task_id="register_endpoint",
        name="sky-register-ray-vllm-endpoint",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["register-endpoint"],
        env_vars={
            "ENDPOINT_URL":  "{{ ti.xcom_pull(task_ids='wait_for_endpoint', key='return_value') }}",
            "SERVE_RUN_ID":  "{{ dag_run.conf['serve_run_id'] }}",
            "HF_MODEL_ID":   "{{ dag_run.conf['llm_model_id'] }}",
            # cluster_name extracted from launch task xcom for the payload
            "CLUSTER_NAME":  "{{ (ti.xcom_pull(task_ids='launch_cluster', key='return_value') or {}).get('cluster_name', '') }}",
            "ORCHESTRATION": "ray_vllm_multinode",
            "S3_BUCKET":     S3_BUCKET,
        },
        env_from=_aws_env_from(),
        get_logs=True,
        is_delete_operator_pod=True,
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "128Mi"},
            limits={"cpu": "200m", "memory": "256Mi"},
        ),
    )

    launch_task >> wait_task >> register_task
