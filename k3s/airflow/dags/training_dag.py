"""Training Pipeline DAG — Ray-only training run.

Triggered by POST /api/v2/runs via dag_run.conf with:
    train_run_id            : str  — unique training run ID
    preprocess_run_id       : str  — referenced preprocessing run ID
    dataset                 : str  — canonical dataset name
    processed_table         : str  — Iceberg processed table to train on
    dsl_s3_path             : str  — S3 URI to DSL YAML
    train_params_s3_path    : str  — S3 URI of params_training.yaml
    model_type              : str  — "xgboost" or "pytorch"

Tasks:
    1. submit_ray_job  — load base YAML, patch + submit RayJob, push name to XCom
    2. poll_ray_job    — poll status, always cleanup in finally
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

try:
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook as _KubernetesHook  # noqa: F401
    _PROVIDERS_AVAILABLE = True
except ImportError:
    _PROVIDERS_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────

RAY_NAMESPACE = "ray"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://my-mlflow.ray.svc.cluster.local:80",
)
_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/


# ─── Tasks ───────────────────────────────────────────────────────────────────


def _submit_ray_job(**context):
    """Load base RayJob YAML, patch per-run env vars, and submit to K8s.

    Pushes the resource name to XCom so poll_ray_job can clean it up.
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError(
            "apache-airflow-providers-cncf-kubernetes is not installed. "
            "Install it in the Airflow image."
        )

    import copy
    import time
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

    base_yaml_path = Path(
        os.getenv("KUBERAY_JOB_YAML", "/opt/airflow/dags/repo/k3s/kuberay/kuberay-job.yaml")
    )
    with open(base_yaml_path) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    # RayJob names must be ≤ 47 characters (KubeRay operator constraint).
    # Prefix "ray" avoids the duplicate "train-train-..." that occurs because
    # train_run_id already starts with "train-".
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

    # Delete pre-existing resource for idempotent reruns
    delete_ray_job(client, RAY_NAMESPACE, ray_job_name)
    time.sleep(2)

    client.create_namespaced_custom_object(
        "ray.io", "v1", RAY_NAMESPACE, "rayjobs", manifest
    )
    print(f"RayJob {ray_job_name} submitted.")

    # Push name so poll task can clean up
    context["ti"].xcom_push(key="ray_job_name", value=ray_job_name)


def _poll_ray_job(**context):
    """Poll RayJob status. Always deletes the resource in finally."""
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import time
    import sys
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import delete_ray_job

    ray_job_name = context["ti"].xcom_pull(
        task_ids="submit_ray_job", key="ray_job_name"
    )

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
            # KubeRay v1: status.jobStatus is a plain string — "PENDING",
            # "RUNNING", "SUCCEEDED", "FAILED", "STOPPED".
            # (The old .get("jobStatus", {}).get("state", "") chain called
            # .get() on a string once the job was RUNNING, raising AttributeError
            # which silently fell into finally: and deleted the RayJob mid-run.)
            state = obj.get("status", {}).get("jobStatus", "").lower()
            print(f"RayJob {ray_job_name} state: {state}")
            if state == "succeeded":
                print("Ray training completed successfully.")
                return
            if state in ("failed", "stopped"):
                raise RuntimeError(f"RayJob {ray_job_name} ended with state: {state}.")
        raise RuntimeError(f"RayJob {ray_job_name} timed out after {timeout}s.")
    finally:
        print(f"Cleanup: deleting RayJob {ray_job_name}")
        delete_ray_job(client, RAY_NAMESPACE, ray_job_name)


# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline",
    description="Ray-only training pipeline. Triggered by POST /api/v2/runs.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "training"],
) as dag:

    submit_ray = PythonOperator(
        task_id="submit_ray_job",
        python_callable=_submit_ray_job,
    )

    poll_ray = PythonOperator(
        task_id="poll_ray_job",
        python_callable=_poll_ray_job,
    )

    submit_ray >> poll_ray
