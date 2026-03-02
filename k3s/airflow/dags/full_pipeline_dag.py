"""Full ML Pipeline DAG — Spark preprocessing → Ray training.

Triggered via dag_run.conf with the union of preprocessing and training keys:
    preprocess_run_id         : str  — unique preprocessing run ID
    artifact_set_id           : str  — last 6 chars of preprocess_run_id
    train_run_id              : str  — unique training run ID
    dataset                   : str  — canonical dataset name
    raw_table                 : str  — full Iceberg table ref
    dsl_s3_path               : str  — S3 URI to DSL YAML
    preprocess_params_s3_path : str  — S3 URI of params_preprocess.yaml
    train_params_s3_path      : str  — S3 URI of params_training.yaml
    schema_s3_path            : str  — S3 URI of full.yaml schema
    model_type                : str  — "xgboost" or "pytorch"

Tasks (linear):
    submit_spark_preprocess >> poll_spark_preprocess >> submit_ray_job >> poll_ray_job

All cleanup is in the poll tasks' finally blocks — always runs on success, failure, or timeout.
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

_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/

# Import task implementations from the dedicated DAG modules
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_dag import _submit_spark_preprocess, _poll_spark_preprocess  # noqa: E402
from training_dag import _submit_ray_job, _poll_ray_job  # noqa: E402


# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="full_ml_pipeline",
    description=(
        "Full ML pipeline: Spark preprocessing → Ray training. "
        "Triggered by POST /api/v2/runs (full mode)."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "preprocessing", "training", "pipeline"],
) as dag:

    submit_spark = PythonOperator(
        task_id="submit_spark_preprocess",
        python_callable=_submit_spark_preprocess,
    )

    poll_spark = PythonOperator(
        task_id="poll_spark_preprocess",
        python_callable=_poll_spark_preprocess,
    )

    submit_ray = PythonOperator(
        task_id="submit_ray_job",
        python_callable=_submit_ray_job,
    )

    poll_ray = PythonOperator(
        task_id="poll_ray_job",
        python_callable=_poll_ray_job,
    )

    submit_spark >> poll_spark >> submit_ray >> poll_ray
