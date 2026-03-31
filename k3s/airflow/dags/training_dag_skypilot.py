"""Training Pipeline DAG — SkyPilot via ExternalPythonOperator.

Calls sky.* Python API directly from the Airflow scheduler process using a
dedicated /opt/skypilot-venv virtualenv, eliminating the separate
KubernetesPodOperator pod that was required to work around protobuf/grpcio
conflicts with Apache Airflow.

SkyPilot connects to the in-cluster API server via:
    SKYPILOT_API_SERVER_ENDPOINT  (env var, read from env-secret)

Sky YAMLs come from the DAG git-sync repo clone (full repo is checked out):
    /opt/airflow/dags/repo/k3s/sky/

Architecture: single-task lifecycle
    run_sky_job:  submit → stream → poll → cancel-on-failure
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
from airflow.operators.python import ExternalPythonOperator
from airflow.task.trigger_rule import TriggerRule

# ─── Constants ────────────────────────────────────────────────────────────────

# Python interpreter inside the isolated SkyPilot venv (built in Dockerfile)
_SKYPILOT_PYTHON = os.getenv("SKYPILOT_VENV_PYTHON", "/opt/skypilot-venv/bin/python")

# Sky YAMLs come from the DAG git-sync repo clone (full repo checked out under /repo/)
_SKY_YAML_DIR = os.getenv("SKY_YAML_DIR", "/opt/airflow/dags/repo/k3s/sky")

SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_TIMEOUT_SECONDS", "7200"))


# ─── Callables (execute inside /opt/skypilot-venv) ────────────────────────────

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
) -> str:
    """Submit + stream + poll + cancel-on-failure. Runs in /opt/skypilot-venv.

    Returns the job_name string (pushed to XCom for the cleanup safety-net task).
    MLFLOW_TRACKING_URI and AWS_* are read from os.environ (inherited from the
    scheduler pod's extraEnv / env-secret).
    """
    import json
    import os
    import re
    import tempfile
    import time

    import yaml
    import sky

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

    tmp = tempfile.mktemp(suffix=".yaml")
    with open(tmp, "w") as fh:
        yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

    task = sky.Task.from_yaml(tmp)
    envs: dict[str, str] = {
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "PARAMS_S3_PATH":      params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "USE_DEEPSPEED":       use_deepspeed,
        "DEEPSPEED_STAGE":     deepspeed_stage,
    }
    if n_nodes > 1:
        envs["NUM_WORKERS"] = str(n_nodes * 8)
    task.update_envs(envs)

    # ── Job name ──────────────────────────────────────────────────────────────
    slug = re.sub(r"[^a-z0-9-]", "-", f"sky-{train_run_id}".lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    job_name = slug[:40]

    # ── Submit ────────────────────────────────────────────────────────────────
    request_id = sky.jobs.launch(task, name=job_name)
    try:
        sky.stream_and_get(request_id)
    except Exception as exc:
        msg = str(exc)
        if "/api/stream" in msg and (
            "Connection refused" in msg or "Max retries exceeded" in msg
        ):
            print(f"[warn] stream_and_get failed: {exc}. Verifying queue visibility...")
            waited = 0
            visible = False
            while waited <= 120:
                try:
                    result = sky.get(sky.jobs.queue(refresh=True, all_users=True, version=2))
                    records = result[0] if isinstance(result, tuple) else result
                    if records and any(
                        (j if isinstance(j, dict) else j.model_dump()).get("job_name") == job_name
                        for j in records
                    ):
                        print(f"[sky-training] Job '{job_name}' visible in queue.")
                        visible = True
                        break
                except Exception:
                    pass
                time.sleep(10)
                waited += 10
            if not visible:
                raise RuntimeError(
                    f"Job '{job_name}' launch stream failed and not visible in queue."
                ) from exc
        else:
            raise

    print(f"[sky-training] Job launched: {job_name}")

    # ── Poll ──────────────────────────────────────────────────────────────────
    interval, elapsed = 30, 0
    try:
        while elapsed < sky_timeout_seconds:
            time.sleep(interval)
            elapsed += interval
            try:
                result = sky.get(sky.jobs.queue(refresh=True, all_users=True, version=2))
                records = result[0] if isinstance(result, tuple) else result
                if not records:
                    continue
                our = [
                    (j if isinstance(j, dict) else j.model_dump())
                    for j in records
                    if (j if isinstance(j, dict) else j.model_dump()).get("job_name") == job_name
                ]
                if not our:
                    continue
                job = our[-1]
                raw_status = job.get("status")
                status = str(getattr(raw_status, "value", None) or raw_status or "")
                print(f"[{elapsed}s] Status: {status}")
                if "SUCCEEDED" in status.upper():
                    print("[sky-training] Training completed successfully.")
                    return job_name
                if "FAILED" in status.upper() or "CANCELLED" in status.upper():
                    details = "; ".join(
                        f"{k}={job.get(k)}"
                        for k in ("failure_reason", "failure_type", "cluster_name")
                        if job.get(k)
                    )
                    raise RuntimeError(f"Job '{job_name}' terminal: {status}. {details}")
            except RuntimeError:
                raise
            except Exception as poll_exc:
                print(f"[warn] poll error (non-fatal): {poll_exc}")
        raise RuntimeError(f"Job '{job_name}' timed out after {sky_timeout_seconds}s.")
    except Exception:
        print(f"[sky-training] Cancelling job '{job_name}'.")
        try:
            sky.get(sky.jobs.cancel(name=job_name))
        except Exception as cancel_exc:
            print(f"[warn] cancel failed: {cancel_exc}")
        raise

    return job_name  # unreachable; satisfies type checker


def _cleanup_failed_training(job_name: str) -> None:
    """Safety-net: cancel managed job if the main task was killed before cleanup ran."""
    import sky

    if not job_name:
        print("[cleanup] No job_name — nothing to cancel.")
        return
    try:
        sky.get(sky.jobs.cancel(name=job_name))
        print(f"[cleanup] Cancelled job '{job_name}'.")
    except Exception as exc:
        print(f"[cleanup] Cancel attempt for '{job_name}': {exc}")


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline_skypilot",
    description=(
        "SkyPilot managed-jobs training pipeline (ExternalPythonOperator). "
        "Calls sky.* directly from Airflow using /opt/skypilot-venv. "
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

    run_sky = ExternalPythonOperator(
        task_id="run_sky_job",
        python=_SKYPILOT_PYTHON,
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
        },
        do_xcom_push=True,
        execution_timeout=timedelta(seconds=SKY_TIMEOUT_SECONDS + 600),
    )

    # Safety-net: cancel the managed job if Airflow killed the main task before
    # its own exception handler ran.
    cleanup_on_failure = ExternalPythonOperator(
        task_id="cleanup_skypilot_controller_on_failure",
        python=_SKYPILOT_PYTHON,
        python_callable=_cleanup_failed_training,
        op_kwargs={
            "job_name": "{{ ti.xcom_pull(task_ids='run_sky_job') or '' }}",
        },
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    run_sky >> cleanup_on_failure
