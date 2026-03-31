"""Training Pipeline DAG — SkyPilot via PythonOperator + subprocess.

Calls the `sky` CLI directly via subprocess from the Airflow scheduler process.
The sky CLI lives in /opt/skypilot-venv/bin/sky (installed in Dockerfile).

SkyPilot connects to the in-cluster API server via:
    SKYPILOT_API_SERVER_ENDPOINT  (env var, inherited from scheduler pod env-secret)

Sky YAMLs come from the DAG git-sync repo clone (full repo is checked out):
    /opt/airflow/dags/repo/k3s/sky/

Architecture: single-task lifecycle
    run_sky_job:  patch YAML → launch → poll → cancel-on-failure
    cleanup_skypilot_controller_on_failure: safety-net cancel (ONE_FAILED)

dag_run.conf keys:
    train_run_id            : str        — unique training run ID
    preprocess_run_id       : str        — referenced preprocessing run ID
    dataset                 : str        — canonical dataset name
    processed_table         : str        — Iceberg processed table to train on
    train_params_s3_path    : str        — S3 URI of params_training.yaml
    model_type              : str        — "xgboost" → CPU  |  "pytorch/ssm/bae" → GPU
    num_nodes               : int        — number of nodes (default 1)
    use_deepspeed           : bool       — install + enable DeepSpeed (default False)
    deepspeed_stage         : int        — ZeRO stage 1|2|3 (default 1)
    resource_constraints    : dict|None  — optional GPU constraints
                                           (must include gpu_fallbacks: [{infra, accelerators, use_spot}])
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.operators.python import PythonOperator
from airflow.task.trigger_rule import TriggerRule

# ─── Constants ────────────────────────────────────────────────────────────────

# sky CLI inside the isolated SkyPilot venv (built in Dockerfile)
_SKY_BIN = os.getenv("SKYPILOT_VENV_BIN", "/opt/skypilot-venv/bin/sky")

# Sky YAMLs come from the DAG git-sync repo clone (full repo checked out under /repo/)
_SKY_YAML_DIR = os.getenv("SKY_YAML_DIR", "/opt/airflow/dags/repo/k3s/sky")

SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_TIMEOUT_SECONDS", "7200"))


# ─── Callables ────────────────────────────────────────────────────────────────

def _run_sky_training(
    train_run_id: str,
    preprocess_run_id: str,
    processed_table: str,
    model_type: str,
    params_s3_path: str,
    num_nodes: str,
    use_deepspeed: str,
    deepspeed_stage: str,
    resource_constraints_json: str,
    sky_timeout_seconds: int,
    sky_yaml_dir: str,
    sky_bin: str,
) -> str:
    """Patch YAML → launch → poll → cancel-on-failure. Returns job_name (XCom).

    MLFLOW_TRACKING_URI and AWS_* are inherited from os.environ (scheduler pod env-secret).
    """
    import json
    import os
    import re
    import subprocess
    import tempfile
    import time

    import yaml

    n_nodes = int(num_nodes)

    rc: dict | None = None
    if resource_constraints_json and resource_constraints_json not in ("null", "{}"):
        try:
            rc = json.loads(resource_constraints_json)
        except Exception:
            pass

    # ── YAML routing ──────────────────────────────────────────────────────────
    use_gpu = model_type in ("pytorch", "ssm", "bae")
    provider = "runpod"
    if rc:
        providers = rc.get("providers") or []
        if providers:
            provider = str(providers[0]).lower()

    _yaml_map = {
        ("train",       "runpod"): "ray-gpu-training-runpod.yaml",
        ("train",       "vast"):   "ray-gpu-training-vast.yaml",
        ("train_multi", "aws"):    "ray-gpu-multinode-aws.yaml",
    }
    kind = "train_multi" if (use_gpu and n_nodes > 1) else "train"
    yaml_name = (
        _yaml_map.get((kind, provider))
        or _yaml_map.get((kind, "runpod"))
        or "ray-gpu-training-runpod.yaml"
    )
    yaml_path = os.path.join(sky_yaml_dir, yaml_name)
    print(f"[sky-training] YAML: {yaml_path}  (model={model_type} gpu={use_gpu} nodes={n_nodes})")

    # ── Load + patch YAML ─────────────────────────────────────────────────────
    with open(yaml_path) as fh:
        sky_conf = yaml.safe_load(fh)

    if rc:
        gpu_fallbacks = rc.get("gpu_fallbacks")
        if gpu_fallbacks and isinstance(gpu_fallbacks, list):
            sky_conf.setdefault("resources", {})["any_of"] = gpu_fallbacks
            sky_conf["resources"].pop("infra", None)
            print(f"[sky-training] Injecting {len(gpu_fallbacks)} gpu_fallbacks into any_of.")

    # Bake per-run env vars into the YAML so sky launch receives them without --env flags
    sky_conf.setdefault("envs", {}).update({
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "PARAMS_S3_PATH":      params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "USE_DEEPSPEED":       use_deepspeed,
        "DEEPSPEED_STAGE":     deepspeed_stage,
    })
    if n_nodes > 1:
        sky_conf["envs"]["NUM_WORKERS"] = str(n_nodes * 8)

    tmp = tempfile.mktemp(suffix=".yaml")
    with open(tmp, "w") as fh:
        yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

    # ── Job name ──────────────────────────────────────────────────────────────
    slug = re.sub(r"[^a-z0-9-]", "-", f"sky-{train_run_id}".lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    job_name = slug[:40]

    # ── Launch ────────────────────────────────────────────────────────────────
    print(f"[sky-training] Launching job '{job_name}'")
    subprocess.run(
        [sky_bin, "jobs", "launch", "-y", "--name", job_name, tmp],
        check=True,
        env=os.environ,
    )
    print(f"[sky-training] Job launched: {job_name}")

    # ── Poll ──────────────────────────────────────────────────────────────────
    interval, elapsed = 30, 0
    try:
        while elapsed < sky_timeout_seconds:
            time.sleep(interval)
            elapsed += interval
            result = subprocess.run(
                [sky_bin, "jobs", "queue", "--all-users"],
                capture_output=True, text=True, check=False, env=os.environ,
            )
            output = result.stdout
            for line in output.splitlines():
                if job_name not in line:
                    continue
                upper = line.upper()
                if "SUCCEEDED" in upper:
                    print(f"[sky-training] Job '{job_name}' succeeded.")
                    return job_name
                if "FAILED" in upper or "CANCELLED" in upper:
                    raise RuntimeError(
                        f"Job '{job_name}' reached terminal state: {line.strip()}"
                    )
                print(f"[{elapsed}s] {line.strip()}")
        raise RuntimeError(f"Job '{job_name}' timed out after {sky_timeout_seconds}s.")
    except Exception:
        print(f"[sky-training] Cancelling job '{job_name}'.")
        subprocess.run(
            [sky_bin, "jobs", "cancel", "-y", "--name", job_name],
            check=False, env=os.environ,
        )
        raise

    return job_name  # unreachable; satisfies type checker


def _cleanup_failed_training(job_name: str, sky_bin: str) -> None:
    """Safety-net: cancel managed job if the main task was killed before cleanup ran."""
    import subprocess
    import os

    if not job_name:
        print("[cleanup] No job_name — nothing to cancel.")
        return
    print(f"[cleanup] Cancelling job '{job_name}'.")
    subprocess.run(
        [sky_bin, "jobs", "cancel", "-y", "--name", job_name],
        check=False, env=os.environ,
    )


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline_skypilot",
    description=(
        "SkyPilot managed-jobs training pipeline (PythonOperator + subprocess). "
        "Calls sky CLI directly from Airflow using /opt/skypilot-venv/bin/sky. "
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

    run_sky = PythonOperator(
        task_id="run_sky_job",
        python_callable=_run_sky_training,
        op_kwargs={
            "train_run_id":              "{{ dag_run.conf['train_run_id'] }}",
            "preprocess_run_id":         "{{ dag_run.conf['preprocess_run_id'] }}",
            "processed_table":           "{{ dag_run.conf.get('processed_table', '') }}",
            "model_type":                "{{ dag_run.conf.get('model_type', 'xgboost') }}",
            "params_s3_path":            "{{ dag_run.conf['train_params_s3_path'] }}",
            "num_nodes":                 "{{ dag_run.conf.get('num_nodes', 1) }}",
            "use_deepspeed":             "{{ 'true' if dag_run.conf.get('use_deepspeed', False) else 'false' }}",
            "deepspeed_stage":           "{{ dag_run.conf.get('deepspeed_stage', 1) | string }}",
            "resource_constraints_json": "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
            "sky_timeout_seconds":       SKY_TIMEOUT_SECONDS,
            "sky_yaml_dir":              _SKY_YAML_DIR,
            "sky_bin":                   _SKY_BIN,
        },
        do_xcom_push=True,
        execution_timeout=timedelta(seconds=SKY_TIMEOUT_SECONDS + 600),
    )

    # Safety-net: cancel the managed job if Airflow killed the main task before
    # its own exception handler ran.
    cleanup_on_failure = PythonOperator(
        task_id="cleanup_skypilot_controller_on_failure",
        python_callable=_cleanup_failed_training,
        op_kwargs={
            "job_name": "{{ ti.xcom_pull(task_ids='run_sky_job') or '' }}",
            "sky_bin":  _SKY_BIN,
        },
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    run_sky >> cleanup_on_failure
