"""Preprocessing Pipeline DAG — Spark-only preprocessing run.

Triggered by POST /api/v2/processing-runs via dag_run.conf with:
    preprocess_run_id         : str  — unique preprocessing run ID
    artifact_set_id           : str  — last 6 chars of preprocess_run_id (Spark table suffix)
    dataset                   : str  — canonical dataset name
    raw_table                 : str  — full Iceberg table ref (e.g. "iceberg.raw.network_traffic")
    dsl_s3_path               : str  — S3 URI to DSL YAML
    preprocess_params_s3_path : str  — S3 URI of params_preprocess.yaml
    schema_s3_path            : str  — S3 URI of full.yaml schema (from schema_ref.full)

Tasks:
    1. submit_spark_preprocess — load base YAML, patch + submit SparkApplication, push name to XCom
    2. poll_spark_preprocess   — poll status, always cleanup in finally
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

SPARK_NAMESPACE = "spark"
_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/


# ─── Tasks ───────────────────────────────────────────────────────────────────


def _submit_spark_preprocess(**context):
    """Load base SparkApplication YAML, patch per-run env vars, and submit to K8s.

    Pushes the resource name to XCom so poll_spark_preprocess can clean it up.
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

    # Allow importing k8s_helpers from the dags repo
    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name, delete_spark_app

    conf = context["dag_run"].conf or {}
    preprocess_run_id = conf["preprocess_run_id"]
    artifact_set_id = conf.get("artifact_set_id", preprocess_run_id[-6:])
    raw_table = conf["raw_table"]
    dsl_s3_path = conf["dsl_s3_path"]
    preprocess_params_s3_path = conf["preprocess_params_s3_path"]
    schema_s3_path = conf.get("schema_s3_path", "")

    base_yaml_path = Path(
        os.getenv("SPARK_APP_YAML", "/opt/airflow/dags/repo/k3s/spark/spark-application.yaml")
    )
    with open(base_yaml_path) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    spark_app_name = k8s_name(preprocess_run_id, "spark-preprocess")
    manifest["metadata"]["name"] = spark_app_name

    per_run_env = {
        "PREPROCESS_RUN_ID": preprocess_run_id,
        "ARTIFACT_SET_ID":   artifact_set_id,
        "RAW_TABLE":         raw_table,
        "DSL_S3_PATH":       dsl_s3_path,
        "PARAMS_S3_PATH":    preprocess_params_s3_path,
    }
    if schema_s3_path:
        per_run_env["SCHEMA_S3_PATH"] = schema_s3_path

    spark_conf = manifest["spec"].setdefault("sparkConf", {})
    for key, val in per_run_env.items():
        spark_conf[f"spark.kubernetes.driverEnv.{key}"] = val
        spark_conf[f"spark.kubernetes.executorEnv.{key}"] = val

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    # Delete pre-existing resource for idempotent reruns
    delete_spark_app(client, SPARK_NAMESPACE, spark_app_name)
    time.sleep(2)

    client.create_namespaced_custom_object(
        "spark.apache.org", "v1", SPARK_NAMESPACE, "sparkapplications", manifest
    )
    print(f"SparkApplication {spark_app_name} submitted.")

    # Push name so poll task can clean up
    context["ti"].xcom_push(key="spark_app_name", value=spark_app_name)


def _poll_spark_preprocess(**context):
    """Poll SparkApplication status. Always deletes the resource in finally."""
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import time
    import sys
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import delete_spark_app

    spark_app_name = context["ti"].xcom_pull(
        task_ids="submit_spark_preprocess", key="spark_app_name"
    )

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    timeout = int(os.getenv("SPARK_TIMEOUT_SECONDS", "1800"))
    interval = 15
    elapsed = 0

    try:
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            obj = client.get_namespaced_custom_object(
                "spark.apache.org", "v1", SPARK_NAMESPACE, "sparkapplications", spark_app_name
            )
            state = (
                obj.get("status", {})
                   .get("currentState", {})
                   .get("currentStateSummary", "")
            )
            print(f"SparkApplication {spark_app_name} state: {state}")
            if state in ("ResourceReleased", "RESOURCE_RELEASED"):
                print("Spark preprocessing completed successfully.")
                return
            if state in ("FAILED", "Failed"):
                raise RuntimeError(f"SparkApplication {spark_app_name} failed.")
        raise RuntimeError(f"SparkApplication {spark_app_name} timed out after {timeout}s.")
    finally:
        print(f"Cleanup: deleting SparkApplication {spark_app_name}")
        delete_spark_app(client, SPARK_NAMESPACE, spark_app_name)


# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="preprocessing_pipeline",
    description="Spark-only preprocessing pipeline. Triggered by POST /api/v2/processing-runs.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "preprocessing"],
) as dag:

    submit_spark = PythonOperator(
        task_id="submit_spark_preprocess",
        python_callable=_submit_spark_preprocess,
    )

    poll_spark = PythonOperator(
        task_id="poll_spark_preprocess",
        python_callable=_poll_spark_preprocess,
    )

    submit_spark >> poll_spark
