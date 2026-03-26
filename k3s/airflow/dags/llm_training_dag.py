"""LLM Training Pipeline DAG — SkyPilot via KubernetesPodOperator.

SkyPilot has a dependency conflict with apache-airflow (protobuf, grpcio).
All sky.* calls are delegated to the takenking9879/sky-runner:0.12.0 pod.

dag_run.conf keys:
    llm_train_run_id        : str        — unique LLM training run ID
    llm_model_id            : str        — HuggingFace model ID (e.g. Qwen/Qwen2.5-7B)
    train_dataset_s3        : str        — s3://bucket/path/train.jsonl
    hf_token                : str        — HuggingFace token (leave empty for ungated models)
    max_steps               : int        — training steps (default 500)
    save_steps              : int        — checkpoint frequency (default 100)
    lora_enabled            : bool       — enable LoRA (default true)
    lora_rank               : int        — LoRA rank (default 8)
    deepspeed_stage         : int        — ZeRO stage 1|2|3 (default 3)
    resource_constraints    : dict|None  — optional GPUSelectorService constraints

Tasks:
    1. submit_llm_job  — sky-runner pod → sky.jobs.launch(); pushes job_name to XCom
    2. poll_llm_job    — sky-runner pod → polls sky.jobs.queue() until terminal state
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
SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_LLM_TIMEOUT_SECONDS", "28800"))  # 8 h

# ─── Shared pod helpers ────────────────────────────────────────────────────────

def _aws_env_from() -> list:
    # Injects AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY,
    # VASTAI_API_KEY, MLFLOW_TRACKING_URI from env-secret (.env).
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="env-secret"))]


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="llm_training_pipeline",
    description=(
        "SkyPilot managed-jobs LLM fine-tuning (KubernetesPodOperator). "
        "HuggingFace TRL SFTTrainer + PEFT LoRA + DeepSpeed ZeRO. "
        "Spot-first with S3 checkpoint recovery."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "llm", "skypilot"],
) as dag:

    submit_llm = KubernetesPodOperator(
        task_id="submit_llm_job",
        name="sky-submit-llm",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["submit-llm"],
        env_vars={
            "LLM_TRAIN_RUN_ID":           "{{ dag_run.conf['llm_train_run_id'] }}",
            "LLM_MODEL_ID":               "{{ dag_run.conf['llm_model_id'] }}",
            "TRAIN_DATASET_S3":           "{{ dag_run.conf['train_dataset_s3'] }}",
            "HF_TOKEN":                   "{{ dag_run.conf.get('hf_token', '') }}",
            "MAX_STEPS":                  "{{ dag_run.conf.get('max_steps', 500) }}",
            "SAVE_STEPS":                 "{{ dag_run.conf.get('save_steps', 100) }}",
            "LORA_ENABLED":               "{{ dag_run.conf.get('lora_enabled', true) | lower }}",
            "LORA_RANK":                  "{{ dag_run.conf.get('lora_rank', 8) }}",
            "DEEPSPEED_STAGE":            "{{ dag_run.conf.get('deepspeed_stage', 3) }}",
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

    poll_llm = KubernetesPodOperator(
        task_id="poll_llm_job",
        name="sky-poll-llm",
        namespace=_AIRFLOW_NS,
        image=_SKY_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["python", "/app/sky_runner.py"],
        arguments=["poll-llm"],
        env_vars={
            "SKY_JOB_NAME":        "{{ ti.xcom_pull(task_ids='submit_llm_job', key='return_value') }}",
            "SKY_TIMEOUT_SECONDS": str(SKY_TIMEOUT_SECONDS),
        },
        env_from=_aws_env_from(),
        get_logs=True,
        is_delete_operator_pod=True,
        execution_timeout=timedelta(seconds=SKY_TIMEOUT_SECONDS + 300),
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "250m", "memory": "512Mi"},
        ),
    )

    submit_llm >> poll_llm
