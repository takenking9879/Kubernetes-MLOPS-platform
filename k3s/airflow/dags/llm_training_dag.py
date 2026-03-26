"""LLM Training Pipeline DAG — SkyPilot managed-jobs LLM fine-tuning.

Uses sky.jobs.launch() for managed spot recovery (auto-restarts on preemption).
Checkpoints are synced to S3 by llm_main.py's S3CheckpointCallback, so
restarts resume from the latest checkpoint rather than from step 0.

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
    1. submit_llm_job   — load sky YAML, inject per-run env vars, launch managed job
    2. poll_llm_job     — poll sky.jobs.queue() every 60s; cancel on timeout/failure

Requirements in Airflow image:
    pip install "skypilot[runpod,aws,vast]" boto3

SkyPilot config on Airflow worker (~/.sky/config.yaml):
    runpod:
      api_key: rpa_XXXXXXXXXXXXXXXXXX
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── Constants ────────────────────────────────────────────────────────────────

# Provider-specific LLM training YAMLs (each uses its own cached base template).
# RunPod and Vast.ai: provider templates (torch pre-installed).
# AWS: custom image docker:takenking9879/ray-llm:2.53.0 (all deps baked in).
SKY_YAML_LLM_RUNPOD = Path(
    os.getenv(
        "SKY_LLM_RUNPOD_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-llm-training-runpod.yaml",
    )
)
SKY_YAML_LLM_VAST = Path(
    os.getenv(
        "SKY_LLM_VAST_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-llm-training-vast.yaml",
    )
)
SKY_YAML_LLM_AWS = Path(
    os.getenv(
        "SKY_LLM_AWS_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-llm-training-aws.yaml",
    )
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://my-mlflow.ray.svc.cluster.local:80",
)
SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_LLM_TIMEOUT_SECONDS", "28800"))  # 8 h

_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _llm_yaml(resource_constraints_conf: dict | None) -> str:
    """Return the correct LLM training YAML path based on target provider."""
    providers = (resource_constraints_conf or {}).get("providers") or []
    p = str(providers[0]).lower() if providers else "runpod"
    if p == "vast":
        return str(SKY_YAML_LLM_VAST)
    if p == "aws":
        return str(SKY_YAML_LLM_AWS)
    return str(SKY_YAML_LLM_RUNPOD)


def _load_llm_task(
    yaml_path: str,
    run_id: str,
    resource_constraints_conf: dict | None,
) -> "sky.Task":  # type: ignore[name-defined]
    """Load a SkyPilot Task, optionally replacing any_of with a dynamic list."""
    import sky  # type: ignore[import]
    import yaml as _yaml

    if not resource_constraints_conf:
        return sky.Task.from_yaml(yaml_path)

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    try:
        import dataclasses
        from src.services.gpu_catalog import GPUCatalogService
        from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

        _valid = {f.name for f in dataclasses.fields(ResourceConstraints)}
        merged = {k: v for k, v in resource_constraints_conf.items() if k in _valid}
        constraints = ResourceConstraints(**merged)

        offers = GPUCatalogService().query_availability(
            providers=constraints.providers,
            min_vram_gb=constraints.min_vram_gb,
            gpu_types=constraints.gpu_types,
        )
        result = GPUSelectorService().select_providers(constraints, offers)

        if not result.any_of:
            print("Dynamic selector empty — falling back to static YAML")
            return sky.Task.from_yaml(yaml_path)

        with open(yaml_path) as fh:
            sky_conf = _yaml.safe_load(fh)

        sky_conf.setdefault("resources", {})["any_of"] = result.any_of
        sky_conf["resources"].pop("infra", None)

        tmp = f"/tmp/sky_llm_{run_id}.yaml"
        with open(tmp, "w") as fh:
            _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

        print(
            f"Dynamic any_of: {len(result.any_of)} entries "
            f"({result.spot_entries} spot, {result.ondemand_entries} on-demand)"
        )
        return sky.Task.from_yaml(tmp)

    except Exception as exc:
        print(f"[warn] Dynamic resource selection failed ({exc}) — using static YAML")
        return sky.Task.from_yaml(yaml_path)


# ─── Tasks ────────────────────────────────────────────────────────────────────


def _submit_llm_job(**context):
    """Load the LLM YAML, inject run env vars, launch a managed job."""
    import sky  # type: ignore[import]

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name  # type: ignore[import]

    conf = context["dag_run"].conf or {}
    run_id = conf["llm_train_run_id"]
    model_id = conf["llm_model_id"]
    dataset_s3 = conf["train_dataset_s3"]
    hf_token = conf.get("hf_token", "")
    max_steps = int(conf.get("max_steps", 500))
    save_steps = int(conf.get("save_steps", 100))
    lora_enabled = str(conf.get("lora_enabled", True)).lower()
    lora_rank = int(conf.get("lora_rank", 8))
    ds_stage = int(conf.get("deepspeed_stage", 3))

    job_name = k8s_name(run_id, "llm", max_len=40)

    resource_constraints_conf = conf.get("resource_constraints")
    yaml_path = _llm_yaml(resource_constraints_conf)
    print(f"Using LLM SkyPilot YAML: {yaml_path}")

    task = _load_llm_task(
        yaml_path,
        run_id,
        resource_constraints_conf,
    )

    task.update_envs({
        "LLM_TRAIN_RUN_ID":     run_id,
        "LLM_MODEL_ID":         model_id,
        "HF_TOKEN":             hf_token,
        "TRAIN_DATASET_S3":     dataset_s3,
        "MAX_STEPS":            str(max_steps),
        "SAVE_STEPS":           str(save_steps),
        "LORA_ENABLED":         lora_enabled,
        "LORA_RANK":            str(lora_rank),
        "DEEPSPEED_STAGE":      str(ds_stage),
        "MLFLOW_TRACKING_URI":  MLFLOW_TRACKING_URI,
    })

    sky.stream_and_get(sky.jobs.launch(
        task,
        name=job_name,
    ))
    print(f"LLM managed job launched: {job_name}")
    context["ti"].xcom_push(key="sky_job_name", value=job_name)


def _poll_llm_job(**context):
    """Poll sky.jobs.queue() every 60s until the LLM job reaches a terminal state."""
    import sky  # type: ignore[import]
    import time

    job_name = context["ti"].xcom_pull(
        task_ids="submit_llm_job", key="sky_job_name"
    )
    print(f"Polling LLM managed job: {job_name}")

    interval = 60  # LLM jobs are long; poll less frequently
    elapsed = 0

    try:
        while elapsed < SKY_TIMEOUT_SECONDS:
            time.sleep(interval)
            elapsed += interval

            try:
                all_jobs = sky.get(sky.jobs.queue(refresh=False))
            except Exception as exc:
                print(f"[{elapsed}s] sky.jobs.queue() error: {exc} — retrying...")
                continue

            our_jobs = [j for j in all_jobs if j.get("job_name") == job_name]
            if not our_jobs:
                print(f"[{elapsed}s] Job '{job_name}' not visible yet (provisioning...)")
                continue

            last = our_jobs[-1]
            status = str(last.get("status", ""))
            print(f"[{elapsed}s] LLM job '{job_name}' status: {status}")

            if "SUCCEEDED" in status:
                print("LLM training completed successfully.")
                return
            if "FAILED" in status or "CANCELLED" in status:
                raise RuntimeError(
                    f"LLM job '{job_name}' ended with status: {status}"
                )
            # PENDING / STARTING / RUNNING / RECOVERING → keep polling

        raise RuntimeError(
            f"LLM job '{job_name}' timed out after {SKY_TIMEOUT_SECONDS}s."
        )

    except Exception:
        print(f"Cancelling LLM job '{job_name}' due to error or timeout.")
        try:
            sky.get(sky.jobs.cancel(name=job_name))
        except Exception as exc:
            print(f"Warning: sky.jobs.cancel('{job_name}') failed: {exc}")
        raise


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="llm_training_pipeline",
    description=(
        "SkyPilot managed-jobs LLM fine-tuning. "
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

    submit_llm = PythonOperator(
        task_id="submit_llm_job",
        python_callable=_submit_llm_job,
    )

    poll_llm = PythonOperator(
        task_id="poll_llm_job",
        python_callable=_poll_llm_job,
    )

    submit_llm >> poll_llm
