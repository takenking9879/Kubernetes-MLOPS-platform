"""Training Pipeline DAG — SkyPilot via PythonOperator + subprocess.

Calls the `sky` CLI directly via subprocess from the Airflow scheduler process.
The sky CLI lives in /opt/skypilot-venv/bin/sky (installed in Dockerfile).

SkyPilot connects to the in-cluster API server via:
    SKYPILOT_API_SERVER_ENDPOINT  (env var, inherited from scheduler pod env-secret)

Sky YAMLs come from the DAG git-sync repo clone (full repo is checked out):
    /opt/airflow/dags/repo/k3s/sky/

Architecture: single-task lifecycle
    run_sky_job:  patch YAML → launch → poll/cancel depending on launch mode
    cleanup_skypilot_controller_on_failure: safety-net cleanup (ONE_FAILED)

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
    use_managed_jobs        : bool       — true -> sky jobs launch, false -> sky launch --retry-until-up
    resource_constraints    : dict|None  — optional GPU constraints
                                           (must include gpu_fallbacks: [{infra, accelerators, use_spot}])
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.task.trigger_rule import TriggerRule

# ─── Constants ────────────────────────────────────────────────────────────────

# sky CLI inside the isolated SkyPilot venv (built in Dockerfile)
_SKY_BIN = os.getenv("SKYPILOT_VENV_BIN", "/opt/skypilot-venv/bin/sky")

# Sky YAMLs come from the DAG git-sync repo clone (full repo checked out under /repo/)
_SKY_YAML_DIR = os.getenv("SKY_YAML_DIR", "/opt/airflow/dags/repo/k3s/sky")

SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_TIMEOUT_SECONDS", "7200"))

# Managed jobs are prone to controller-state races in some SkyPilot versions
# under heavy recovery. Default to direct `sky launch` to avoid that path.
SKY_USE_MANAGED_JOBS = os.getenv("SKY_USE_MANAGED_JOBS", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# ─── Callables ────────────────────────────────────────────────────────────────

def _run_sky_training(
    train_run_id: str,
    airflow_run_id: str,
    airflow_task_try_number: str,
    preprocess_run_id: str,
    processed_table: str,
    model_type: str,
    params_s3_path: str,
    num_nodes: str,
    num_gpus_per_node: str,
    use_deepspeed: str,
    deepspeed_stage: str,
    resource_constraints_json: str,
    sky_timeout_seconds: int,
    sky_yaml_dir: str,
    sky_bin: str,
    use_managed_jobs: str,
) -> str:
    """Patch YAML and run training via SkyPilot.

    Returns:
        Resource name pushed to XCom:
        - managed mode: managed job name
        - direct mode: cluster name

    MLFLOW_TRACKING_URI is forwarded as a SkyPilot secret when present.
    Sensitive values (AWS credentials, etc.) are forwarded via `sky jobs launch --secret`
    so secret values are not written to the generated YAML file.
    """
    import json
    import os
    import re
    import subprocess
    import tempfile
    import time
    from urllib.parse import urlparse

    import yaml

    n_nodes = int(num_nodes)

    # Sky jobs do not automatically inherit scheduler env vars into the remote
    # training VM. Forward sensitive values via SkyPilot secrets.
    aws_access_key_id = (os.environ.get("AWS_ACCESS_KEY_ID") or "").strip()
    aws_secret_access_key = (os.environ.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    aws_session_token = (os.environ.get("AWS_SESSION_TOKEN") or "").strip()
    grafana_remote_write_url = (os.environ.get("GRAFANA_REMOTE_WRITE_URL") or "").strip()
    grafana_instance_id = (os.environ.get("GRAFANA_INSTANCE_ID") or "").strip()
    grafana_api_key = (os.environ.get("GRAFANA_API_KEY") or "").strip()
    metrics_remote_write_url = (os.environ.get("METRICS_REMOTE_WRITE_URL") or "").strip()
    metrics_remote_write_username = (os.environ.get("METRICS_REMOTE_WRITE_USERNAME") or "").strip()
    metrics_remote_write_password = (os.environ.get("METRICS_REMOTE_WRITE_PASSWORD") or "").strip()
    mlflow_tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    aws_region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-2"
    ).strip()
    aws_s3_endpoint_url = (
        os.environ.get("AWS_S3_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL")
        or ""
    ).strip()

    if not aws_access_key_id or not aws_secret_access_key:
        raise RuntimeError(
            "Missing AWS credentials in Airflow scheduler environment. "
            "Expected AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env-secret."
        )

    secret_env_keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    if aws_session_token:
        secret_env_keys.append("AWS_SESSION_TOKEN")
    if metrics_remote_write_password:
        secret_env_keys.append("METRICS_REMOTE_WRITE_PASSWORD")
    if grafana_api_key:
        secret_env_keys.append("GRAFANA_API_KEY")
    if mlflow_tracking_uri:
        secret_env_keys.append("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        parsed_mlflow = urlparse(mlflow_tracking_uri)
        mlflow_host = parsed_mlflow.netloc or parsed_mlflow.path or "<unknown>"
        print(
            "[sky-training] Airflow env MLFLOW_TRACKING_URI detected "
            f"(host={mlflow_host}); forwarding via --secret."
        )
    else:
        print(
            "[sky-training] Airflow env MLFLOW_TRACKING_URI is empty; "
            "remote job will use params_training.yaml fallback."
        )

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

    effective_metrics_remote_write_url = metrics_remote_write_url or grafana_remote_write_url
    effective_metrics_remote_write_username = metrics_remote_write_username or grafana_instance_id
    if effective_metrics_remote_write_url:
        rw_host = urlparse(effective_metrics_remote_write_url).netloc or "<unknown>"
        print(
            "[sky-training] Outbound metrics shipping enabled "
            f"(remote_write_host={rw_host}, provider={provider})."
        )
    else:
        print("[sky-training] Outbound metrics shipping disabled (missing remote_write URL).")

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
        explicit_gpu_selection = bool(
            (rc.get("gpu_types") or [])
            or (rc.get("preferred_regions") or [])
            or (rc.get("gpu_fallbacks") or [])
        )
        if gpu_fallbacks and isinstance(gpu_fallbacks, list):
            resources_cfg = sky_conf.setdefault("resources", {})
            resources_cfg["ordered"] = gpu_fallbacks
            resources_cfg.pop("any_of", None)
            resources_cfg.pop("infra", None)
            print(f"[sky-training] Injecting {len(gpu_fallbacks)} gpu_fallbacks into ordered.")
        elif explicit_gpu_selection:
            raise RuntimeError(
                "resource_constraints included explicit GPU/region selections but "
                "gpu_fallbacks is empty; refusing static YAML fallback."
            )

    # Bake non-sensitive per-run env vars into YAML.
    sky_envs = sky_conf.setdefault("envs", {})
    sky_envs.update({
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "SKY_PROVIDER":        provider,
        "METRICS_SOURCE":      "skypilot_vm",
        "METRICS_REMOTE_WRITE_URL": effective_metrics_remote_write_url,
        "METRICS_REMOTE_WRITE_USERNAME": effective_metrics_remote_write_username,
        # Keep legacy aliases for templates still using Grafana Cloud names.
        "GRAFANA_REMOTE_WRITE_URL": effective_metrics_remote_write_url,
        "GRAFANA_INSTANCE_ID": effective_metrics_remote_write_username,
        "PARAMS_S3_PATH":      params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "AWS_REGION":          aws_region,
        "AWS_DEFAULT_REGION":  aws_region,
        "AWS_S3_ENDPOINT_URL": aws_s3_endpoint_url,
        "USE_DEEPSPEED":       use_deepspeed,
        "DEEPSPEED_STAGE":     deepspeed_stage,
    })

    # Remove secret placeholders from template envs to avoid persisting secret keys
    # with empty/default values in the generated YAML.
    for secret_key in secret_env_keys:
        sky_envs.pop(secret_key, None)

    # Declare required secrets by name (values still passed via --secret CLI flags).
    sky_secrets = sky_conf.get("secrets")
    if not isinstance(sky_secrets, dict):
        sky_secrets = {}
        sky_conf["secrets"] = sky_secrets
    for secret_key in secret_env_keys:
        sky_secrets[secret_key] = None

    # Compute GPU-derived Ray parameters and inject into YAML envs.
    # Primary source of truth: num_nodes (from dag_run.conf) × num_gpus_per_node (from resource_constraints).
    # The run script in the YAML has a secondary fallback using SKYPILOT_* runtime vars.
    n_gpus = int(num_gpus_per_node) if str(num_gpus_per_node).strip().isdigit() else 1
    total_gpus = n_nodes * n_gpus

    # Worker strategy:
    # - single-node: workers = total GPUs
    # - multi-node: workers = total nodes
    # Keep training/tuning shape equal on fixed-size clusters.
    if n_nodes > 1:
        worker_count = n_nodes
    else:
        worker_count = total_gpus

    num_workers_training = worker_count
    # Tune trials use a single worker; parallelism is controlled by MAX_CONCURRENT_TRIALS.
    num_workers_tune = 1
    max_concurrent_trials = worker_count

    sky_envs["NUM_WORKERS"] = str(num_workers_training)
    sky_envs["NUM_WORKERS_TUNE"] = str(num_workers_tune)
    sky_envs["MAX_CONCURRENT_TRIALS"] = str(max_concurrent_trials)
    print(
        f"[sky-training] GPU resource plan: {n_nodes} node(s) × {n_gpus} GPU(s) = {total_gpus} total | "
        f"training workers={num_workers_training} | "
        f"tune workers_per_trial={num_workers_tune} | concurrent_trials={max_concurrent_trials}"
    )

    # ── Job name ──────────────────────────────────────────────────────────────
    # Keep managed job names unique per Airflow run attempt so cancel/recovery
    # from one run cannot accidentally target another run with the same
    # train_run_id.
    slug = re.sub(r"[^a-z0-9-]", "-", f"sky-{train_run_id}".lower())
    slug = re.sub(r"-+", "-", slug).strip("-") or "sky-training"

    run_marker = f"{airflow_run_id}-{airflow_task_try_number}".strip("-")
    if not run_marker:
        run_marker = f"{time.time_ns()}-{os.getpid()}"

    import hashlib

    suffix = hashlib.sha1(run_marker.encode("utf-8")).hexdigest()[:8]
    managed_mode = str(use_managed_jobs).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if managed_mode:
        base_max_len = max(1, 40 - len(suffix) - 1)
    else:
        # Keep cluster names slightly shorter for cloud/provider compatibility.
        base_max_len = max(1, 32 - len(suffix) - 1)

    base = slug[:base_max_len].rstrip("-") or "sky"
    resource_name = f"{base}-{suffix}"
    sky_envs["SKY_CLUSTER_NAME"] = resource_name

    tmp = tempfile.mktemp(suffix=".yaml")
    with open(tmp, "w") as fh:
        yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

    if managed_mode:
        # ── Managed jobs mode (optional) ─────────────────────────────────────
        launch_cmd = [sky_bin, "jobs", "launch", "-y", "--name", resource_name, tmp]
        for secret_key in secret_env_keys:
            launch_cmd.extend(["--secret", secret_key])

        print(
            f"[sky-training] Launching MANAGED job '{resource_name}' with secrets: "
            + ", ".join(secret_env_keys)
        )
        subprocess.run(
            launch_cmd,
            check=True,
            env=os.environ,
        )
        print(f"[sky-training] Managed job launched: {resource_name}")

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
                    if resource_name not in line:
                        continue
                    upper = line.upper()
                    if "SUCCEEDED" in upper:
                        print(f"[sky-training] Job '{resource_name}' succeeded.")
                        return resource_name
                    if "FAILED" in upper or "CANCELLED" in upper:
                        raise RuntimeError(
                            f"Job '{resource_name}' reached terminal state: {line.strip()}"
                        )
                    print(f"[{elapsed}s] {line.strip()}")
            raise RuntimeError(
                f"Job '{resource_name}' timed out after {sky_timeout_seconds}s."
            )
        except Exception:
            print(f"[sky-training] Cancelling managed job '{resource_name}'.")
            subprocess.run(
                [sky_bin, "jobs", "cancel", "-y", "--name", resource_name],
                check=False,
                env=os.environ,
            )
            raise

    # ── Direct launch mode (default; avoids managed-jobs controller races) ───
    launch_cmd = [
        sky_bin,
        "launch",
        "-y",
        "--retry-until-up",
        "-c",
        resource_name,
        tmp,
    ]
    for secret_key in secret_env_keys:
        launch_cmd.extend(["--secret", secret_key])

    print(
        f"[sky-training] Launching DIRECT cluster '{resource_name}' with secrets: "
        + ", ".join(secret_env_keys)
    )

    keep_cluster = os.getenv("SKY_KEEP_CLUSTER_AFTER_TRAINING", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    try:
        # Blocking run; returns when task command exits.
        subprocess.run(launch_cmd, check=True, env=os.environ)
        print(f"[sky-training] Direct run completed on cluster: {resource_name}")
        return resource_name
    finally:
        if keep_cluster:
            print(
                f"[sky-training] Keeping cluster '{resource_name}' "
                "(SKY_KEEP_CLUSTER_AFTER_TRAINING=true)."
            )
        else:
            subprocess.run(
                [sky_bin, "down", "-y", resource_name],
                check=False,
                env=os.environ,
            )


def _cleanup_failed_training(job_name: str, sky_bin: str) -> None:
    """Safety-net cleanup for both managed and direct launch modes."""
    import subprocess
    import os

    if not job_name:
        print("[cleanup] No resource name — nothing to cleanup.")
        return

    print(f"[cleanup] Attempting managed-job cancel for '{job_name}' (best effort).")
    subprocess.run(
        [sky_bin, "jobs", "cancel", "-y", "--name", job_name],
        check=False, env=os.environ,
    )
    print(f"[cleanup] Attempting cluster down for '{job_name}' (best effort).")
    subprocess.run(
        [sky_bin, "down", "-y", job_name],
        check=False,
        env=os.environ,
    )


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline_skypilot",
    description=(
        "SkyPilot training pipeline (PythonOperator + subprocess). "
        "Defaults to direct sky launch to avoid managed-jobs controller races; "
        "managed-jobs mode can be re-enabled with SKY_USE_MANAGED_JOBS=true. "
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
            "airflow_run_id":            "{{ run_id }}",
            "airflow_task_try_number":   "{{ ti.try_number | string }}",
            "preprocess_run_id":         "{{ dag_run.conf['preprocess_run_id'] }}",
            "processed_table":           "{{ dag_run.conf.get('processed_table', '') }}",
            "model_type":                "{{ dag_run.conf.get('model_type', 'xgboost') }}",
            "params_s3_path":            "{{ dag_run.conf['train_params_s3_path'] }}",
            "num_nodes":                 "{{ dag_run.conf.get('num_nodes', 1) }}",
            "num_gpus_per_node":         "{{ dag_run.conf.get('num_gpus_per_node', 1) }}",
            "use_deepspeed":             "{{ 'true' if dag_run.conf.get('use_deepspeed', False) else 'false' }}",
            "deepspeed_stage":           "{{ dag_run.conf.get('deepspeed_stage', 1) | string }}",
            "resource_constraints_json": "{{ (dag_run.conf.get('resource_constraints') or {}) | tojson }}",
            "sky_timeout_seconds":       SKY_TIMEOUT_SECONDS,
            "sky_yaml_dir":              _SKY_YAML_DIR,
            "sky_bin":                   _SKY_BIN,
            "use_managed_jobs":          "{{ dag_run.conf.get('use_managed_jobs', params.sky_use_managed_jobs) }}",
        },
        do_xcom_push=True,
        execution_timeout=timedelta(seconds=SKY_TIMEOUT_SECONDS + 600),
        params={
            "sky_use_managed_jobs": SKY_USE_MANAGED_JOBS,
        },
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
