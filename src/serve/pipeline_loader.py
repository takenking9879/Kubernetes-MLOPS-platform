"""
Pipeline artifact loader for online serving mode.

Resolves and downloads DSL pipeline artifacts through the same chain
used by kafka_main.py — without any Spark dependency:

  MLflow alias (champion/challenger)
      └─► artifact_set_id tag on the registered model version
            └─► pyiceberg: metadata.preprocessing_artifacts → pipeline_hash
                  └─► S3: {bucket}/pipelines/{pipeline_hash}/stages.json
                              {bucket}/pipelines/{pipeline_hash}/config.json
                        └─► NumpyPipelineExecutor

Used by app.py when kuberay.serving.online = true.
"""

import logging
import os
import tempfile
import time
from typing import Optional

import yaml


class PipelineArtifactLoader:
    """
    Resolves pipeline artifacts from MLflow → Iceberg → S3 and loads
    a NumpyPipelineExecutor.

    Mirrors kafka_main.py's artifact-loading logic without Spark:
      - _resolve_artifact_set_id_from_mlflow  → MLflow MlflowClient
      - _resolve_pipeline_hash_from_metadata  → pyiceberg (not Spark SQL)
      - _download_artifacts                   → boto3
    """

    def __init__(self, params_path: str, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
        with open(params_path) as f:
            self._params = yaml.safe_load(f)

    # ── Public API ───────────────────────────────────────────────────────────────

    def load_executor(self, registry_name: str, alias: str):
        """
        Full resolution chain: MLflow → Iceberg → S3 → NumpyPipelineExecutor.

        Args:
            registry_name: MLflow Model Registry name (e.g. "attack-detection")
            alias:         Model alias to resolve (e.g. "champion", "challenger")

        Returns:
            NumpyPipelineExecutor loaded from the resolved pipeline artifacts.

        Raises:
            RuntimeError / FileNotFoundError on any resolution failure.
        """
        from src.dsl.numpy_executor import NumpyPipelineExecutor

        total_started = time.perf_counter()

        t0 = time.perf_counter()
        artifact_set_id = self._resolve_artifact_set_id(registry_name, alias)
        t1 = time.perf_counter()

        pipeline_hash   = self._resolve_pipeline_hash(artifact_set_id)
        t2 = time.perf_counter()

        tmpdir          = tempfile.mkdtemp(prefix="pipeline_artifacts_")

        self._download_artifacts(pipeline_hash, tmpdir)
        t3 = time.perf_counter()

        executor = NumpyPipelineExecutor.from_dir(tmpdir)
        t4 = time.perf_counter()

        self._logger.info(
            "NumpyPipelineExecutor ready: %d features (artifact_set_id=%s, pipeline_hash=%s)",
            len(executor.final_features),
            artifact_set_id,
            pipeline_hash,
        )
        self._logger.info(
            "Pipeline loader timings (s): mlflow=%.2f iceberg=%.2f s3=%.2f executor=%.2f total=%.2f",
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t4 - total_started,
        )
        return executor

    # ── Step 1: MLflow → artifact_set_id ────────────────────────────────────────

    def _resolve_artifact_set_id(self, registry_name: str, alias: str) -> str:
        """
        Reads the 'artifact_set_id' tag from the MLflow model version
        identified by registry_name@alias.

        Mirrors kafka_main.py's _resolve_artifact_set_id_from_mlflow().
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self._params.get("kuberay", {}).get("model", {}).get("mlflow_tracking_uri"),
        )
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        mv   = client.get_model_version_by_alias(registry_name, alias)
        tags = getattr(mv, "tags", {}) or {}
        artifact_set_id = tags.get("artifact_set_id")

        if not artifact_set_id:
            raise RuntimeError(
                f"MLflow model {registry_name!r}@{alias!r} v{mv.version} "
                f"has no tag 'artifact_set_id'. "
                f"Run preprocessing and ensure the model version is tagged correctly."
            )

        self._logger.info(
            "MLflow %s@%s v%s → artifact_set_id=%s",
            registry_name, alias, mv.version, artifact_set_id,
        )
        return str(artifact_set_id)

    # ── Step 2: artifact_set_id → pipeline_hash (via pyiceberg) ─────────────────

    def _resolve_pipeline_hash(self, artifact_set_id: str) -> str:
        """
        Queries the Iceberg metadata table (metadata.preprocessing_artifacts)
        for the pipeline_hash that corresponds to the given artifact_set_id.

        Uses pyiceberg directly — no Spark required.
        Mirrors kafka_main.py's _resolve_pipeline_hash_from_metadata().
        """
        from pyiceberg.catalog import load_catalog
        from pyiceberg.expressions import EqualTo

        catalog_kwargs = self._get_iceberg_catalog_kwargs()
        meta_cfg       = self._params.get("iceberg_tables", {}).get("metadata", {})
        identifier     = (
            f"{meta_cfg.get('namespace', 'metadata')}."
            f"{meta_cfg.get('table', 'preprocessing_artifacts')}"
        )

        self._logger.info(
            "Querying Iceberg table %r for artifact_set_id=%s", identifier, artifact_set_id
        )
        catalog = load_catalog("iceberg", **catalog_kwargs)
        table   = catalog.load_table(identifier)

        # Fast path: predicate pushdown to avoid full-table scan on every replica init.
        try:
            scan = table.scan(
                row_filter=EqualTo("artifact_set_id", artifact_set_id),
                selected_fields=("artifact_set_id", "pipeline_hash"),
            )
            try:
                scan = scan.limit(1)
            except Exception:
                # limit() may not exist on older pyiceberg versions.
                pass
            rows = scan.to_pandas()
        except Exception as exc:
            self._logger.warning(
                "Filtered Iceberg scan failed (%s); falling back to full-table scan.",
                exc,
            )
            df = table.scan().to_pandas()
            rows = df[df["artifact_set_id"] == artifact_set_id]

        # Defensive fallback if pushdown path returns no rows due type/engine mismatch.
        if rows.empty:
            self._logger.warning(
                "Filtered Iceberg scan returned no rows for artifact_set_id=%s; "
                "retrying with full-table scan.",
                artifact_set_id,
            )
            df = table.scan().to_pandas()
            rows = df[df["artifact_set_id"] == artifact_set_id]

        if rows.empty:
            raise RuntimeError(
                f"No row found in Iceberg table {identifier!r} "
                f"for artifact_set_id={artifact_set_id!r}. "
                f"Run preprocessing first."
            )

        pipeline_hash = str(rows.iloc[0]["pipeline_hash"])
        if not pipeline_hash:
            raise RuntimeError(
                f"pipeline_hash is empty for artifact_set_id={artifact_set_id!r} "
                f"in table {identifier!r}."
            )

        self._logger.info(
            "Iceberg %s: artifact_set_id=%s → pipeline_hash=%s",
            identifier, artifact_set_id, pipeline_hash,
        )
        return pipeline_hash

    # ── Step 3: pipeline_hash → download from S3 ────────────────────────────────

    def _download_artifacts(self, pipeline_hash: str, tmpdir: str) -> None:
        """
        Downloads pipeline artifact files from S3:
          s3://{bucket}/{prefix}/{pipeline_hash}/stages.json
          s3://{bucket}/{prefix}/{pipeline_hash}/config.json

        Mirrors kafka_main.py's _download_pipeline_from_key().
        """
        import boto3

        artifacts_cfg = (
            self._params.get("spark", {})
            .get("preprocessing", {})
            .get("artifacts", {})
        )
        bucket  = artifacts_cfg.get("bucket", "k8s-mlops-platform-bucket")
        prefix  = artifacts_cfg.get("dynamic_key_prefix", "pipelines").strip("/")
        archives = artifacts_cfg.get("archives", ["config.json", "stages.json"])

        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
        )

        for archive in archives:
            s3_key     = f"{prefix}/{pipeline_hash}/{archive}"
            local_path = os.path.join(tmpdir, archive)
            self._logger.info("Downloading s3://%s/%s → %s", bucket, s3_key, local_path)
            s3.download_file(bucket, s3_key, local_path)

    # ── Iceberg catalog config (same pattern as kuberay/main.py) ────────────────

    def _get_iceberg_catalog_kwargs(self) -> dict:
        warehouse = (
            self._params.get("iceberg_tables", {})
            .get("warehouse", "s3://k8s-mlops-platform-bucket/warehouse")
            .replace("s3a://", "s3://")
        )
        return {
            "type": "glue",
            "warehouse": warehouse,
            "client.access-key-id":     os.environ.get("AWS_ACCESS_KEY_ID"),
            "client.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "client.region":            os.environ.get("AWS_REGION", "us-east-2"),
        }
