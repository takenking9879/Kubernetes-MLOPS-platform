"""GPU Training Pipeline DAG — external GPU worker (Docker Desktop / RunPod).

Triggered by POST /api/v2/runs via dag_run.conf with:
    train_run_id            : str   — unique training run ID
    preprocess_run_id       : str   — referenced preprocessing run ID
    dataset                 : str   — canonical dataset name
    processed_table         : str   — Iceberg processed table to train on
    dsl_s3_path             : str   — S3 URI to DSL YAML
    train_params_s3_path    : str   — S3 URI of params_training.yaml
    model_type              : str   — "xgboost" or "pytorch"

    # GPU-specific (optional)
    use_runpod              : bool  — True to provision a RunPod pod automatically.
                                      False (default) assumes the GPU worker is already
                                      running locally (Docker Desktop PoC) or pre-started.
    gpu_type                : str   — RunPod GPU type. Default: RUNPOD_GPU_TYPE env var
                                      or "NVIDIA RTX 4090".
    runpod_image            : str   — Docker image for the RunPod pod.
                                      Default: RUNPOD_IMAGE env var.

Tasks:
    1. provision_runpod_gpu  — (only when use_runpod=True) provisions a RunPod pod,
                               waits for it to reach RUNNING, pushes pod_id to XCom.
    2. submit_ray_job_gpu    — submits the RayJob using kuberay-job-gpu-runpod.yaml,
                               pushes ray_job_name to XCom.
    3. poll_ray_job_gpu      — polls RayJob status; in the finally block, terminates
                               the RunPod pod (if provisioned) and deletes the RayJob.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

try:
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook as _KubernetesHook  # noqa: F401
    _PROVIDERS_AVAILABLE = True
except ImportError:
    _PROVIDERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

RAY_NAMESPACE = "ray"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://my-mlflow.ray.svc.cluster.local:80",
)
_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/

# Path to the head-only RayJob manifest (no GPU workers in K8s)
_GPU_RUNPOD_YAML = os.getenv(
    "KUBERAY_JOB_GPU_RUNPOD_YAML",
    "/opt/airflow/dags/repo/k3s/kuberay/kuberay-job-gpu-runpod.yaml",
)

# RunPod defaults (overridden by dag_run.conf or env vars from runpod-secrets)
_DEFAULT_GPU_TYPE = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA RTX 4090")
_DEFAULT_RUNPOD_IMAGE = os.getenv("RUNPOD_IMAGE", "")

# Grace period (seconds) after RunPod pod is RUNNING, before submitting the
# RayJob. Gives the worker time to start Tailscale and join the Ray cluster.
_WORKER_JOIN_GRACE_SECONDS = int(os.getenv("WORKER_JOIN_GRACE_SECONDS", "90"))


# ─── Task 1: Provision RunPod GPU pod ────────────────────────────────────────


def _provision_runpod_gpu(**context):
    """Provision a RunPod GPU pod and wait for it to reach RUNNING.

    Skipped entirely when use_runpod=False (local Docker PoC mode).

    Pushes to XCom:
        runpod_pod_id  — RunPod pod ID string (or "" if not provisioned)
    """
    sys.path.insert(0, str(_DAGS_DIR))
    sys.path.insert(0, str(_DAGS_DIR.parent.parent / "k3s" / "runpod"))

    conf = context["dag_run"].conf or {}
    use_runpod: bool = conf.get("use_runpod", False)

    # Always push so poll task can read it safely even if skipped
    context["ti"].xcom_push(key="runpod_pod_id", value="")

    if not use_runpod:
        logger.info(
            "use_runpod=False — skipping RunPod provisioning. "
            "Ensure a local GPU Docker container is running and has joined the Ray cluster."
        )
        return

    from runpod_helpers import RunPodClient, generate_tailscale_ephemeral_key

    gpu_type = conf.get("gpu_type", _DEFAULT_GPU_TYPE)
    image = conf.get("runpod_image", _DEFAULT_RUNPOD_IMAGE)
    if not image:
        raise ValueError(
            "runpod_image not provided in dag_run.conf and RUNPOD_IMAGE env var is not set. "
            "Build and push the image with: "
            "docker build -f k3s/runpod/Dockerfile.runpod-gpu -t <registry>/ray-gpu-runpod:2.53.0 ."
        )

    # Resolve the Ray head address that the RunPod worker will dial
    # For RunPod (remote), this must be the Tailscale IP/DNS of the head pod.
    # Set RAY_HEAD_TAILSCALE_ADDRESS in your Airflow env or Airflow Variable
    # after setting up the Tailscale Operator (see plan docs).
    ray_head_address = os.getenv("RAY_HEAD_TAILSCALE_ADDRESS", "")
    if not ray_head_address:
        raise ValueError(
            "RAY_HEAD_TAILSCALE_ADDRESS is not set. "
            "Set it to the Tailscale IP or DNS name of the Ray head pod, e.g. "
            "'100.x.x.x:6379' or 'ray-head.<tailnet>.ts.net:6379'."
        )

    # Generate a one-use Tailscale ephemeral key for this pod
    ts_key = generate_tailscale_ephemeral_key()

    runpod = RunPodClient()
    pod_id = runpod.provision(
        image=image,
        gpu_type=gpu_type,
        ray_head_address=ray_head_address,
        tailscale_authkey=ts_key,
    )
    logger.info("RunPod pod provisioned: %s", pod_id)
    context["ti"].xcom_push(key="runpod_pod_id", value=pod_id)

    # Wait for the pod to be RUNNING (typically 1-3 min cold start)
    runpod.wait_for_running(pod_id, timeout=300)
    logger.info(
        "RunPod pod %s is RUNNING. Waiting %ds for worker to join Ray...",
        pod_id,
        _WORKER_JOIN_GRACE_SECONDS,
    )
    time.sleep(_WORKER_JOIN_GRACE_SECONDS)


# ─── Task 2: Submit GPU RayJob ────────────────────────────────────────────────


def _submit_ray_job_gpu(**context):
    """Load the GPU RayJob manifest, patch per-run env vars, and submit to K8s.

    Uses kuberay-job-gpu-runpod.yaml (head only, no GPU worker in K8s).
    The GPU worker was started externally in the previous task.

    Pushes to XCom:
        ray_job_name  — K8s resource name for the submitted RayJob
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError(
            "apache-airflow-providers-cncf-kubernetes is not installed. "
            "Add it to the Airflow image."
        )

    import copy
    import yaml as _yaml
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name, delete_ray_job

    conf = context["dag_run"].conf or {}
    train_run_id = conf["train_run_id"]
    preprocess_run_id = conf["preprocess_run_id"]
    processed_table = conf.get("processed_table", "")
    model_type = conf.get("model_type", "xgboost")
    train_params_s3_path = conf["train_params_s3_path"]

    with open(_GPU_RUNPOD_YAML) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    # RayJob names must be ≤ 47 characters (KubeRay operator constraint)
    ray_job_name = k8s_name(train_run_id, "ray", max_len=47)
    manifest["metadata"]["name"] = ray_job_name

    extra_env = {
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "PARAMS_S3_PATH":      train_params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    }

    existing_runtime_env = _yaml.safe_load(
        manifest["spec"].get("runtimeEnvYAML", "env_vars: {}") or "env_vars: {}"
    )
    merged_env = {**existing_runtime_env.get("env_vars", {}), **extra_env}
    env_lines = "\n".join(f"      {k}: \"{v}\"" for k, v in merged_env.items())
    manifest["spec"]["runtimeEnvYAML"] = f"env_vars:\n{env_lines}\n"

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    delete_ray_job(client, RAY_NAMESPACE, ray_job_name)
    time.sleep(2)

    client.create_namespaced_custom_object(
        "ray.io", "v1", RAY_NAMESPACE, "rayjobs", manifest
    )
    logger.info("RayJob %s submitted (GPU/RunPod variant).", ray_job_name)

    context["ti"].xcom_push(key="ray_job_name", value=ray_job_name)


# ─── Task 3: Poll + cleanup ───────────────────────────────────────────────────


def _poll_ray_job_gpu(**context):
    """Poll RayJob status until completion.

    In the finally block, always:
      - deletes the RayJob resource
      - terminates the RunPod pod (if one was provisioned)
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import sys
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    sys.path.insert(0, str(_DAGS_DIR))
    sys.path.insert(0, str(_DAGS_DIR.parent.parent / "k3s" / "runpod"))
    from k8s_helpers import delete_ray_job
    from runpod_helpers import RunPodClient

    ray_job_name = context["ti"].xcom_pull(
        task_ids="submit_ray_job_gpu", key="ray_job_name"
    )
    runpod_pod_id: str = context["ti"].xcom_pull(
        task_ids="provision_runpod_gpu", key="runpod_pod_id"
    ) or ""

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    timeout = int(os.getenv("RAY_TIMEOUT_SECONDS", "3600"))
    interval = 30
    elapsed = 0

    try:
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            obj = client.get_namespaced_custom_object(
                "ray.io", "v1", RAY_NAMESPACE, "rayjobs", ray_job_name
            )
            state = obj.get("status", {}).get("jobStatus", "").lower()
            logger.info("RayJob %s state: %s (%ds elapsed)", ray_job_name, state, elapsed)
            if state == "succeeded":
                logger.info("GPU training completed successfully.")
                return
            if state in ("failed", "stopped"):
                raise RuntimeError(
                    f"RayJob {ray_job_name} ended with state: {state}."
                )
        raise RuntimeError(
            f"RayJob {ray_job_name} timed out after {timeout}s."
        )
    finally:
        logger.info("Cleanup: deleting RayJob %s", ray_job_name)
        delete_ray_job(client, RAY_NAMESPACE, ray_job_name)

        if runpod_pod_id:
            logger.info("Cleanup: terminating RunPod pod %s", runpod_pod_id)
            try:
                RunPodClient().terminate(runpod_pod_id)
            except Exception as exc:
                # Log but don't re-raise — the RayJob result already determined
                # the task outcome. A termination failure is an ops concern.
                logger.error(
                    "Failed to terminate RunPod pod %s: %s", runpod_pod_id, exc
                )


# ─── DAG definition ──────────────────────────────────────────────────────────


with DAG(
    dag_id="training_pipeline_gpu",
    description=(
        "GPU training pipeline using an external GPU worker "
        "(Docker Desktop local or RunPod remote). "
        "Triggered by POST /api/v2/runs."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "training", "gpu", "runpod"],
) as dag:

    provision_gpu = PythonOperator(
        task_id="provision_runpod_gpu",
        python_callable=_provision_runpod_gpu,
    )

    submit_ray = PythonOperator(
        task_id="submit_ray_job_gpu",
        python_callable=_submit_ray_job_gpu,
    )

    poll_ray = PythonOperator(
        task_id="poll_ray_job_gpu",
        python_callable=_poll_ray_job_gpu,
    )

    provision_gpu >> submit_ray >> poll_ray
