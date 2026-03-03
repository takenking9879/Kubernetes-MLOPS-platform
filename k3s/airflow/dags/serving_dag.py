"""Serving Pipeline DAG — MLflow promotion + RayService patch + optional Spark Kafka connector.

Triggered by POST /api/v2/serving-configs/{serve_run_id}/deploy via dag_run.conf with:
    serve_run_id            : str       — unique serving run ID
    train_run_id            : str       — training run being promoted to champion
    dataset                 : str       — canonical dataset name
    serving_mode            : str       — "ray_only" | "kafka"
    registry_model_name     : str       — MLflow registered model name
    params_serving_s3_path  : str       — S3 URI of params_serving.yaml
    ray_service_name        : str       — K8s RayService name (e.g. "model-serving")
    ray_service_namespace   : str       — K8s namespace (e.g. "ray")
    raw_schema_s3_path      : str|null  — S3 URI of raw.yaml (kafka mode only)

Tasks:
    promote_to_champion
        ↓
    patch_ray_service_config
        ↓
    branch_on_serving_mode  (BranchPythonOperator)
        ├── ray_only → skip_kafka  (EmptyOperator)
        └── kafka    →
            validate_raw_schema
                ↓
            delete_existing_spark_connector
                ↓
            submit_spark_connector
                ↓
            poll_spark_connector_running

Cleanup strategy:
  Spark kafka connector: NOT deleted on success (streaming app must keep running).
  Deleted in finally ONLY when failed=True.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

# ─── Constants ────────────────────────────────────────────────────────────────

SPARK_NAMESPACE = "spark"
RAY_NAMESPACE = "ray"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://my-mlflow.ray.svc.cluster.local:80",
)

_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/

try:
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook as _KubernetesHook  # noqa: F401
    _PROVIDERS_AVAILABLE = True
except ImportError:
    _PROVIDERS_AVAILABLE = False


# ─── Tasks ───────────────────────────────────────────────────────────────────


def _promote_to_champion(**context):
    """Promote the model version tagged with train_run_id to 'champion' alias in MLflow.

    Searches model versions by tag train_run_id (set by training code) and assigns
    the 'champion' alias. Idempotent: set_registered_model_alias is safe to re-run.
    Pushes promoted_version to XCom.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    conf = context["dag_run"].conf or {}
    train_run_id = conf["train_run_id"]
    registry_model_name = conf["registry_model_name"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    versions = client.search_model_versions(
        filter_string=f"name='{registry_model_name}' and tags.train_run_id='{train_run_id}'"
    )

    if not versions:
        raise RuntimeError(
            f"No model version found for registry_model_name='{registry_model_name}' "
            f"with tag train_run_id='{train_run_id}'. "
            "Ensure the training job sets mlflow.set_tag('train_run_id', ...) before registering."
        )

    # If multiple versions share the same train_run_id (e.g. retries), take the highest
    target = max(versions, key=lambda v: int(v.version))
    version_str = target.version

    print(
        f"Promoting {registry_model_name} v{version_str} "
        f"(train_run_id={train_run_id}) to @champion"
    )

    # Idempotent: moves 'champion' from previous version automatically
    client.set_registered_model_alias(
        name=registry_model_name,
        alias="champion",
        version=version_str,
    )

    print(f"Champion alias set: {registry_model_name}@champion → v{version_str}")
    context["ti"].xcom_push(key="promoted_version", value=version_str)


def _patch_ray_service_config(**context):
    """Create or patch the RayService to inject PARAMS_SERVING_S3_PATH.

    Strategy:
      - If the RayService exists: read serveConfigV2, inject PARAMS_SERVING_S3_PATH
        into every application's runtime_env.env_vars, and MERGE-patch.
      - If the RayService does NOT exist (first deploy): load the base manifest from
        the repo, inject PARAMS_SERVING_S3_PATH, and CREATE the resource.

    KubeRay detects the spec change and triggers a rolling reload of Ray Serve
    deployments. ConfigLoader.load() picks up the new PARAMS_SERVING_S3_PATH at
    startup, loading the new params_serving.yaml for alias and canary config.
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import copy
    import yaml as _yaml
    from kubernetes import client as _k8s_api
    from kubernetes.client.exceptions import ApiException
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    conf = context["dag_run"].conf or {}
    params_serving_s3_path = conf["params_serving_s3_path"]
    ray_service_name = conf.get("ray_service_name", "model-serving")
    ray_service_namespace = conf.get("ray_service_namespace", RAY_NAMESPACE)

    hook = KubernetesHook(conn_id="kubernetes_default")
    k8s_client = _k8s_api.CustomObjectsApi(hook.get_conn())

    def _inject_env(serve_config: dict) -> dict:
        updated = copy.deepcopy(serve_config)
        for app in updated.get("applications", []):
            runtime_env = app.setdefault("runtime_env", {})
            env_vars = runtime_env.setdefault("env_vars", {})
            env_vars["PARAMS_SERVING_S3_PATH"] = params_serving_s3_path
            print(
                f"Injecting PARAMS_SERVING_S3_PATH into app '{app.get('name', '?')}': "
                f"{params_serving_s3_path}"
            )
        return updated

    try:
        obj = k8s_client.get_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=ray_service_namespace,
            plural="rayservices",
            name=ray_service_name,
        )
        # ── Existing resource: patch serveConfigV2 ──────────────────────────
        serve_config_str = obj.get("spec", {}).get("serveConfigV2", "")
        serve_config = _yaml.safe_load(serve_config_str) or {}
        updated = _inject_env(serve_config)
        patch_body = {
            "spec": {
                "serveConfigV2": _yaml.dump(updated, default_flow_style=False),
            }
        }
        k8s_client.patch_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=ray_service_namespace,
            plural="rayservices",
            name=ray_service_name,
            body=patch_body,
        )
        print(
            f"RayService '{ray_service_name}' patched: "
            f"PARAMS_SERVING_S3_PATH={params_serving_s3_path}"
        )

    except ApiException as exc:
        if exc.status != 404:
            raise
        # ── First deploy: create from base manifest ─────────────────────────
        import os as _os
        from pathlib import Path as _Path

        base_yaml_path = _Path(
            _os.getenv(
                "RAYSERVICE_YAML",
                "/opt/airflow/dags/repo/k3s/kuberay/serving/rayservice-model-serving.yaml",
            )
        )
        if not base_yaml_path.exists():
            raise FileNotFoundError(
                f"RayService '{ray_service_name}' not found in cluster and base manifest "
                f"not found at {base_yaml_path}. Deploy the RayService manually first or "
                "set RAYSERVICE_YAML to the correct path."
            ) from exc

        with open(base_yaml_path) as fh:
            manifest = _yaml.safe_load(fh)

        serve_config_str = manifest.get("spec", {}).get("serveConfigV2", "")
        serve_config = _yaml.safe_load(serve_config_str) or {}
        updated = _inject_env(serve_config)
        manifest["spec"]["serveConfigV2"] = _yaml.dump(updated, default_flow_style=False)

        k8s_client.create_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=ray_service_namespace,
            plural="rayservices",
            body=manifest,
        )
        print(
            f"RayService '{ray_service_name}' created (first deploy): "
            f"PARAMS_SERVING_S3_PATH={params_serving_s3_path}"
        )


def _branch_on_serving_mode(**context):
    """Return task_id of next task based on serving_mode from conf."""
    conf = context["dag_run"].conf or {}
    serving_mode = conf.get("serving_mode", "ray_only")
    if serving_mode == "kafka":
        return "validate_raw_schema"
    return "skip_kafka"


def _validate_raw_schema(**context):
    """Validate that raw_schema_s3_path is set and the S3 object exists.

    Fails fast before submitting Spark, avoiding a launch that fails 2 minutes
    later due to a missing schema file.
    """
    import re
    import boto3
    from botocore.exceptions import ClientError

    conf = context["dag_run"].conf or {}
    raw_schema_s3_path = conf.get("raw_schema_s3_path") or ""

    if not raw_schema_s3_path:
        raise ValueError(
            "serving_mode is 'kafka' but raw_schema_s3_path is not set in dag_run.conf. "
            "Re-submit the serving config with a saved raw.yaml "
            "(POST /api/v2/schemas/{dataset}/raw)."
        )

    match = re.match(r"s3://([^/]+)/(.+)", raw_schema_s3_path)
    if not match:
        raise ValueError(f"raw_schema_s3_path is not a valid S3 URI: {raw_schema_s3_path}")

    bucket, key = match.group(1), match.group(2)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-2"),
    )
    try:
        s3.head_object(Bucket=bucket, Key=key)
        print(f"raw.yaml validated at {raw_schema_s3_path}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            raise RuntimeError(
                f"raw.yaml not found at {raw_schema_s3_path}. "
                "Save the schema via POST /api/v2/schemas/{dataset}/raw first."
            ) from e
        raise


def _delete_existing_spark_connector(**context):
    """Idempotent pre-delete of any existing Spark Kafka connector for this serve run.

    Uses k8s_name() to derive the resource name (same helper as preprocessing_dag.py).
    Swallows 404 — safe to call even when no connector exists.
    Pushes spark_app_name to XCom for downstream tasks.
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import time
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name, delete_spark_app

    conf = context["dag_run"].conf or {}
    serve_run_id = conf["serve_run_id"]

    spark_app_name = k8s_name(serve_run_id, "spark-kafka")

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    delete_spark_app(client, SPARK_NAMESPACE, spark_app_name)
    time.sleep(2)

    context["ti"].xcom_push(key="spark_app_name", value=spark_app_name)
    print(f"Pre-delete complete: {spark_app_name}")


def _submit_spark_connector(**context):
    """Submit the Spark Kafka streaming SparkApplication.

    Loads the base YAML from SPARK_KAFKA_YAML env var, patches metadata.name
    and injects per-run env vars via spark.kubernetes.driverEnv.* sparkConf
    entries. Mirrors exactly the pattern in _submit_spark_preprocess().
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import copy
    import yaml as _yaml
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    conf = context["dag_run"].conf or {}
    serve_run_id = conf["serve_run_id"]
    params_serving_s3_path = conf["params_serving_s3_path"]
    raw_schema_s3_path = conf.get("raw_schema_s3_path") or ""

    spark_app_name = context["ti"].xcom_pull(
        task_ids="delete_existing_spark_connector", key="spark_app_name"
    )

    base_yaml_path = Path(
        os.getenv(
            "SPARK_KAFKA_YAML",
            "/opt/airflow/dags/repo/k3s/spark/inference/spark-kafka-application.yaml",
        )
    )
    with open(base_yaml_path) as fh:
        manifest = copy.deepcopy(_yaml.safe_load(fh))

    manifest["metadata"]["name"] = spark_app_name

    per_run_env = {
        "SERVE_RUN_ID": serve_run_id,
        "PARAMS_SERVING_S3_PATH": params_serving_s3_path,
        "RAW_SCHEMA_S3_PATH": raw_schema_s3_path,
    }

    spark_conf = manifest["spec"].setdefault("sparkConf", {})
    for key, val in per_run_env.items():
        spark_conf[f"spark.kubernetes.driverEnv.{key}"] = val
        spark_conf[f"spark.kubernetes.executorEnv.{key}"] = val

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    client.create_namespaced_custom_object(
        "spark.apache.org", "v1", SPARK_NAMESPACE, "sparkapplications", manifest
    )
    print(f"SparkApplication {spark_app_name} submitted (streaming — will not terminate).")

    context["ti"].xcom_push(key="spark_app_name", value=spark_app_name)


def _poll_spark_connector_running(**context):
    """Poll SparkApplication until state == RUNNING (streaming app stays up).

    Key difference from _poll_spark_preprocess:
      - Polls until currentStateSummary is RUNNING, NOT ResourceReleased
      - Does NOT delete the resource on success (streaming app must keep running)
      - Deletes ONLY if failed=True in the finally block

    On RUNNING for STABLE_RUNNING_SECONDS (30s) consecutively: return success.
    App keeps running after DAG completes.
    """
    if not _PROVIDERS_AVAILABLE:
        raise RuntimeError("apache-airflow-providers-cncf-kubernetes is not installed.")

    import time
    from kubernetes import client as _k8s_api
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import delete_spark_app

    spark_app_name = context["ti"].xcom_pull(
        task_ids="submit_spark_connector", key="spark_app_name"
    )

    hook = KubernetesHook(conn_id="kubernetes_default")
    client = _k8s_api.CustomObjectsApi(hook.get_conn())

    timeout = int(os.getenv("SPARK_KAFKA_TIMEOUT_SECONDS", "600"))
    interval = 15
    elapsed = 0
    running_since: float | None = None
    STABLE_RUNNING_SECONDS = 30
    failed = False

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
            print(f"SparkApplication {spark_app_name} state: {state} (elapsed={elapsed}s)")

            if state in ("RUNNING", "Running"):
                if running_since is None:
                    running_since = time.time()
                    print(
                        f"Connector entered RUNNING state — confirming stability "
                        f"for {STABLE_RUNNING_SECONDS}s"
                    )
                elif time.time() - running_since >= STABLE_RUNNING_SECONDS:
                    print(
                        f"SparkApplication {spark_app_name} confirmed RUNNING for "
                        f"{STABLE_RUNNING_SECONDS}s — streaming connector is healthy."
                    )
                    return  # SUCCESS — do not delete

            elif state in ("ResourceReleased", "RESOURCE_RELEASED"):
                failed = True
                raise RuntimeError(
                    f"SparkApplication {spark_app_name} terminated unexpectedly "
                    f"(state: {state}). A streaming app should never complete normally. "
                    "Check Spark driver logs."
                )
            elif state in ("FAILED", "Failed"):
                failed = True
                raise RuntimeError(
                    f"SparkApplication {spark_app_name} failed (state: {state})."
                )
            else:
                # PENDING, SUBMITTED, etc. — reset stability timer if we drop out of RUNNING
                if running_since is not None:
                    print(
                        f"Connector left RUNNING state (now '{state}') — "
                        "resetting stability timer"
                    )
                    running_since = None

        failed = True
        raise RuntimeError(
            f"SparkApplication {spark_app_name} did not reach RUNNING state "
            f"within {timeout}s."
        )

    finally:
        if failed:
            print(f"Cleanup: deleting failed SparkApplication {spark_app_name}")
            delete_spark_app(client, SPARK_NAMESPACE, spark_app_name)
        else:
            print(
                f"Skipping cleanup: {spark_app_name} is a streaming app "
                "and must keep running."
            )


# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="serving_pipeline",
    description=(
        "Serving pipeline: MLflow promotion → RayService patch → optional Spark Kafka connector. "
        "Triggered by POST /api/v2/serving-configs/{serve_run_id}/deploy."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "serving"],
) as dag:

    promote = PythonOperator(
        task_id="promote_to_champion",
        python_callable=_promote_to_champion,
    )

    patch_ray = PythonOperator(
        task_id="patch_ray_service_config",
        python_callable=_patch_ray_service_config,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_serving_mode",
        python_callable=_branch_on_serving_mode,
    )

    skip_kafka = EmptyOperator(task_id="skip_kafka")

    validate_schema = PythonOperator(
        task_id="validate_raw_schema",
        python_callable=_validate_raw_schema,
    )

    delete_connector = PythonOperator(
        task_id="delete_existing_spark_connector",
        python_callable=_delete_existing_spark_connector,
    )

    submit_connector = PythonOperator(
        task_id="submit_spark_connector",
        python_callable=_submit_spark_connector,
    )

    poll_connector = PythonOperator(
        task_id="poll_spark_connector_running",
        python_callable=_poll_spark_connector_running,
    )

    # Task dependency graph
    promote >> patch_ray >> branch
    branch >> skip_kafka
    branch >> validate_schema >> delete_connector >> submit_connector >> poll_connector
