"""ML Pipeline DAG — UI-triggered end-to-end training pipeline.

Triggered by the UI via dag_run.conf with:
    execution_id   : str  — unique run identifier (e.g. "20260223_120000")
    raw_table      : str  — full Iceberg table ref (e.g. "iceberg.raw.network_traffic_raw")
    dsl_s3_path    : str  — S3 URI to DSL YAML (e.g. "s3://bucket/dsl/dsl_network_traffic/v1.yaml")
    params_s3_path : str  — S3 URI of execution params.yaml
    model_type     : str  — "xgboost" or "pytorch"
    schema_s3_path : str  — (optional) S3 URI to schema YAML; auto-derived from raw_table if absent

Tasks:
    1. preprocess — SparkKubernetesOperator: fit DSL + write processed Iceberg table
    2. train      — KubernetesPodOperator (kubectl):  Ray distributed training + MLflow logging

After successful training the UI can optionally trigger a serving deployment.
The existing promotion_dag.py handles champion/challenger alias assignment.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Provider imports with graceful fallback for environments where providers
# are not yet installed (e.g. local lint/test).
# ---------------------------------------------------------------------------
try:
    from airflow.providers.cncf.kubernetes.operators.spark_kubernetes import (
        SparkKubernetesOperator,
    )
    from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
        KubernetesPodOperator,
    )
    from kubernetes.client import V1EnvVar
    _PROVIDERS_AVAILABLE = True
except ImportError:
    _PROVIDERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants — override via Airflow Variables or env vars in the Airflow pod
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.mlflow.svc.cluster.local:5000",
)
S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
SPARK_NAMESPACE = "spark"
RAY_NAMESPACE = "ray"
SPARK_APP_NAME = "spark-app-pipeline"
RAY_JOB_NAME = "kuberay-job-pipeline"


def _submit_spark_preprocessing(**context):
    """Load the base SparkApplication YAML, patch name + per-run env vars, and submit.

    Env vars are injected via sparkConf (spark.kubernetes.driverEnv.* /
    spark.kubernetes.executorEnv.*) — the same mechanism already used for
    PYTHONPATH and PROMETHEUS_PORT in spark-application.yaml.

    Mirrors what a manual kubectl apply of spark-application.yaml does,
    preserving the full cluster spec (sparkConf, driverSpec, executorSpec,
    initContainers, volumes, etc.) without duplicating it in the DAG.
    """
    conf = context["dag_run"].conf or {}
    execution_id = conf.get("execution_id", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    raw_table = conf.get("raw_table", "")
    dsl_s3_path = conf.get("dsl_s3_path", "")
    params_s3_path = conf.get("params_s3_path", "")
    schema_s3_path = conf.get("schema_s3_path", "")

    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError(
            "apache-airflow-providers-cncf-kubernetes is not installed. "
            "Install it in the Airflow image."
        )

    import copy
    import yaml as _yaml
    from pathlib import Path
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
    import time

    # Load the base SparkApplication manifest — same YAML a manual kubectl apply uses.
    # The full cluster spec (sparkConf, driverSpec, executorSpec, initContainers,
    # volumes, git-sync) lives there; we only patch the name and sparkConf env vars.
    base_yaml_path = Path(
        os.getenv("SPARK_APP_YAML", "/git-data/repo/k3s/spark/spark-application.yaml")
    )
    with open(base_yaml_path) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    # Unique name per execution (avoids conflicts with manual kubectl runs)
    manifest["metadata"]["name"] = f"{SPARK_APP_NAME}-{execution_id}"

    # Inject per-run env vars via sparkConf — identical to how PYTHONPATH is set.
    # spark.kubernetes.driverEnv.KEY  → visible to the driver Python process
    # spark.kubernetes.executorEnv.KEY → visible to executor Python processes
    per_run_env = {
        "EXECUTION_ID":   execution_id,
        "ARTIFACT_SET_ID": execution_id,
        "RAW_TABLE":      raw_table,
        "DSL_S3_PATH":    dsl_s3_path,
        "PARAMS_S3_PATH": params_s3_path,
    }
    if schema_s3_path:
        per_run_env["SCHEMA_S3_PATH"] = schema_s3_path

    spark_conf = manifest["spec"].setdefault("sparkConf", {})
    for key, val in per_run_env.items():
        spark_conf[f"spark.kubernetes.driverEnv.{key}"] = val
        spark_conf[f"spark.kubernetes.executorEnv.{key}"] = val

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = hook.get_custom_object_client()

    name = manifest["metadata"]["name"]
    namespace = SPARK_NAMESPACE
    group = "spark.apache.org"
    version = "v1"
    plural = "sparkapplications"

    # Delete any pre-existing resource with the same name (idempotent reruns)
    try:
        client.delete_namespaced_custom_object(group, version, namespace, plural, name)
        time.sleep(3)
    except Exception:
        pass

    client.create_namespaced_custom_object(group, version, namespace, plural, manifest)
    context["task_instance"].log.info(f"SparkApplication {name} submitted.")

    # Poll until ResourceReleased or FAILED
    timeout = 1800
    interval = 15
    elapsed = 0
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval
        obj = client.get_namespaced_custom_object(group, version, namespace, plural, name)
        state = (
            obj.get("status", {})
               .get("currentState", {})
               .get("currentStateSummary", "")
        )
        context["task_instance"].log.info(f"SparkApplication state: {state}")
        if state in ("ResourceReleased", "RESOURCE_RELEASED"):
            context["task_instance"].log.info("Spark preprocessing completed.")
            return
        if state in ("FAILED", "Failed"):
            raise RuntimeError(f"SparkApplication {name} failed.")

    raise RuntimeError(f"SparkApplication {name} timed out after {timeout}s.")


def _submit_ray_training(**context):
    """Patch and apply RayJob manifest via kubectl in a KubernetesPodOperator-style call.

    Uses the Kubernetes API to apply a RayJob with execution env vars injected
    into runtimeEnvYAML. Mirrors what training.sh does for manual runs.
    """
    conf = context["dag_run"].conf or {}
    execution_id = conf.get("execution_id", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    model_type = conf.get("model_type", "xgboost")
    params_s3_path = conf.get("params_s3_path", "")

    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError(
            "apache-airflow-providers-cncf-kubernetes is not installed."
        )

    import copy
    import yaml as _yaml
    from pathlib import Path
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
    import time

    # Load the base RayJob manifest (same YAML training.sh applies manually).
    # The full cluster spec (head/worker groups, images, volumes) lives there —
    # we only patch the name and runtimeEnvYAML, nothing else.
    base_yaml_path = Path(
        os.getenv("KUBERAY_JOB_YAML", "/git-data/repo/k3s/kuberay/kuberay-job.yaml")
    )
    with open(base_yaml_path) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    # Unique name per execution (avoids conflicts with manual training.sh runs)
    manifest["metadata"]["name"] = f"{RAY_JOB_NAME}-{execution_id}"

    # Patch runtimeEnvYAML: merge existing env vars with per-run overrides
    extra_env = {
        "EXECUTION_ID":      execution_id,
        "MODEL_TYPE":        model_type,
        "PARAMS_S3_PATH":    params_s3_path,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    }
    existing_runtime_env = _yaml.safe_load(
        manifest["spec"].get("runtimeEnvYAML", "env_vars: {}") or "env_vars: {}"
    )
    merged_env = {**existing_runtime_env.get("env_vars", {}), **extra_env}
    env_lines = "\n".join(f"      {k}: \"{v}\"" for k, v in merged_env.items())
    manifest["spec"]["runtimeEnvYAML"] = f"env_vars:\n{env_lines}\n"

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = hook.get_custom_object_client()
    name = manifest["metadata"]["name"]
    group = "ray.io"
    version = "v1"
    plural = "rayjobs"

    try:
        client.delete_namespaced_custom_object(group, version, RAY_NAMESPACE, plural, name)
        time.sleep(3)
    except Exception:
        pass

    client.create_namespaced_custom_object(group, version, RAY_NAMESPACE, plural, manifest)
    context["task_instance"].log.info(f"RayJob {name} submitted.")

    # Poll until succeeded or failed
    timeout = 3600
    interval = 30
    elapsed = 0
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval
        obj = client.get_namespaced_custom_object(group, version, RAY_NAMESPACE, plural, name)
        state = (
            obj.get("status", {})
               .get("jobStatus", {})
               .get("state", "")
        ).lower()
        context["task_instance"].log.info(f"RayJob state: {state}")
        if state == "succeeded":
            context["task_instance"].log.info("Ray training completed successfully.")
            return
        if state == "failed":
            raise RuntimeError(f"RayJob {name} failed.")

    raise RuntimeError(f"RayJob {name} timed out after {timeout}s.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="ml_pipeline",
    description=(
        "UI-triggered ML pipeline: Spark preprocessing → Ray training. "
        "Triggered via dag_run.conf with execution_id, raw_table, dsl_s3_path, "
        "params_s3_path, model_type."
    ),
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "training", "pipeline"],
) as dag:

    preprocess = PythonOperator(
        task_id="spark_preprocess",
        python_callable=_submit_spark_preprocessing,
    )

    train = PythonOperator(
        task_id="ray_train",
        python_callable=_submit_ray_training,
    )

    preprocess >> train
