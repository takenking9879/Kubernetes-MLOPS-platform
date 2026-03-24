"""KubeRay Training Orchestrator.

Uses the BaseTrainer / BaseTuner framework to dispatch training to the
correct framework implementation without ``if framework == ...`` branching.
"""

from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import ray
import re
import os
import pickle
import subprocess
import yaml
import time
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import GreaterThanOrEqual, LessThanOrEqual, And
from urllib.parse import urlparse
from typing import Dict, Any
from ray.train.xgboost import RayTrainReportCallback

from src.schemas.model.pytorch_params import PYTORCH_PARAMS
from src.schemas.model.xgboost_params import XGBOOST_PARAMS
from src.pipeline.utils.mlflow_utils import log_training_run
from src.schemas.spark.schema_registry import SchemaRegistry

# ===== FRAMEWORK REGISTRIES =====
from src.pipeline.train import get_trainer
from src.pipeline.tuning import get_tuner

# ===== PROMETHEUS METRICS =====
from prometheus_client import start_http_server
from src.prometheus import (
    TRAIN_FAILURES,
    TRAIN_SPLIT_ROWS,
    PrometheusTuneCallback,
    export_final_metrics,
)


# ===== DEFAULT PARAMS REGISTRY =====
# Maps framework name → (default_params_dict, params_key)
_FRAMEWORK_DEFAULTS: Dict[str, tuple] = {
    "pytorch": (PYTORCH_PARAMS, "pytorch_params"),
    "xgboost": (XGBOOST_PARAMS, "xgboost_params"),
}


class KubeRayTraining(BaseUtils):
    def __init__(self, params_path: str, output_dir: str):
        logger = create_logger('KubeRayTraining', 'kuberay_training.log')
        super().__init__(logger, params_path)
        self.params_full = self.load_params()
        self.params = self.params_full['kuberay']['model']
        self.schema = self._load_schema_features()
        self.output_dir = output_dir

        # Airflow injects TRAIN_RUN_ID as an env var per RayJob submission so
        # that multiple concurrent/sequential runs each carry their own ID.
        # We surface it into params_full so every method reads from one place.
        # Fallback chain: env var → execution.train_run_id (set by backend) → run_metadata.run_id
        env_train_run_id = (
            os.getenv("TRAIN_RUN_ID")
            or self.params_full.get("execution", {}).get("train_run_id", "")
            or self.params_full.get("run_metadata", {}).get("run_id", "")
        )
        if env_train_run_id:
            self.params_full.setdefault("run_metadata", {})["train_run_id"] = env_train_run_id

        # Iceberg setup
        self.catalog_kwargs = self._get_iceberg_catalog()

        # Start Prometheus metrics server
        self._start_prometheus_server()

    def _get_iceberg_catalog(self):
        """Build Iceberg catalog kwargs for Ray's ``read_iceberg``."""
        # Prefer iceberg_tables.warehouse (set in params_training.yaml and static params.yaml).
        # Fall back to hardcoded default if neither key is present.
        warehouse = (
            self.params_full.get('iceberg_tables', {}).get('warehouse')
            or self.params_full.get('spark', {}).get('iceberg', {}).get('warehouse')
            or 's3a://k8s-mlops-platform-bucket/warehouse'
        ).replace('s3a://', 's3://')

        self.logger.info(f"Using Iceberg warehouse: {warehouse}")
        return {
            "type": "glue",
            "warehouse": warehouse,
            "client.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "client.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "client.region": os.environ.get("AWS_REGION", "us-east-2"),
        }

    def _get_latest_artifact(self):
        """Retrieves the latest (artifact_set_id, processed_table_full) from the metadata table.

        Returns a tuple (artifact_set_id, processed_table_name) where processed_table_name
        is the full Iceberg reference stored by the preprocessing job
        (e.g. 'iceberg.processed.network_traffic_raw_20260223_050000').
        Returns (None, None) if no metadata records exist.
        """
        try:
            meta_cfg = self.params_full.get('iceberg_tables', {}).get('metadata', {})
            identifier = f"{meta_cfg.get('namespace', 'metadata')}.{meta_cfg.get('table', 'preprocessing_artifacts')}"

            table = load_catalog("iceberg", **self.catalog_kwargs).load_table(identifier)
            df = table.scan().to_pandas()
            if df.empty:
                return None, None

            latest = df.sort_values('created_at', ascending=False).iloc[0]
            artifact_set_id = latest['artifact_set_id']
            processed_table_full = latest['processed_table_name']
            self.logger.info(
                f"Latest artifact found — id={artifact_set_id}, table={processed_table_full}"
            )
            return artifact_set_id, processed_table_full
        except Exception as e:
            self.logger.warning(f"Could not find latest artifact from metadata: {e}")
            return None, None

    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server on port 8002."""
        port = int(os.getenv('PROMETHEUS_PORT', 8002))
        try:
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")
        except OSError as e:
            if "Address already in use" in str(e):
                self.logger.info(f"Prometheus server already running on port {port}")
        except Exception as e:
            self.logger.error(f"Could not start Prometheus server: {e}")
            raise

    def _check_minio_connection(self):
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-2'),
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            self.s3 = s3
            self.logger.info('Minio connection verified. Buckets: %s', bucket_names)
        except Exception as e:
            self.logger.error('Minio connection failed: %s', str(e), exc_info=True)
            raise

    def _load_data(self, split: str, table_name: str):
        """Loads a specific split from Iceberg using timestamp ranges from params.yaml."""
        try:
            split_cfg = self.params_full.get('splits', {}).get(split, {})
            start_ts = self.format_iceberg_ts(split_cfg.get('start'))
            end_ts = self.format_iceberg_ts(split_cfg.get('end'))

            row_filter = None
            if start_ts and end_ts:
                self.logger.info(f"Loading {split} split from {table_name} ({start_ts} to {end_ts})")
                row_filter = And(
                    GreaterThanOrEqual("timestamp", start_ts),
                    LessThanOrEqual("timestamp", end_ts)
                )
            else:
                self.logger.warning(f"No valid split configuration for {split}. Loading all data.")

            ds = ray.data.read_iceberg(table_identifier=table_name, catalog_kwargs=self.catalog_kwargs, row_filter=row_filter)

            self._validate_schema(ds)

            if os.getenv("RAY_MATERIALIZE_DATASETS", "0") in ("1", "true", "True"):
                self.logger.info(f"Materializing {split} dataset in Ray Object Store for performance.")
                ds = ds.materialize()
            self.logger.info(f"Data for split '{split}' loaded.")
            return ds
        except Exception as e:
            self.logger.error(f"Failed to load {split} data from {table_name}: {str(e)}", exc_info=True)
            raise

    def _validate_schema(self, ds: ray.data.Dataset):
        """Validates dataset schema against Spark preprocessing contract."""
        try:
            cols = set(ds.schema().names)

            if not self.schema:
                self.schema = SchemaRegistry.get_schema('schema_preprocessed')

            expected = {
                self.target_col,
                *self.schema
            }

            missing = expected - cols
            if missing:
                raise ValueError(f"Data Validation failed: missing {list(missing)}")

            self.logger.info(f"Schema validation passed. Total columns: {len(cols)}")
        except Exception as e:
            self.logger.error(str(e))
            raise

    def _save_model(self, result, framework):
        """Extract the trained model from checkpoint and save to S3 as .pkl."""
        self.logger.info(f"Exportando modelo final de {framework} a S3...")
        try:
            checkpoint = result.checkpoint
            if not checkpoint:
                self.logger.warning("No se encontro un checkpoint valido en el resultado.")
                return

            ckpt_path = checkpoint.path
            raw_path = ckpt_path[5:] if ckpt_path.startswith("s3://") else ckpt_path
            bucket_in, prefix_in = raw_path.split("/", 1)

            target_file = "model.pt" if framework == "pytorch" else "model.ubj"
            key_in = f"{prefix_in.rstrip('/')}/{target_file}"

            self.logger.info(f"Descargando {target_file} desde s3://{bucket_in}/{key_in}")
            response = self.s3.get_object(Bucket=bucket_in, Key=key_in)
            model_bytes = response['Body'].read()

            if framework == "xgboost":
                import xgboost as xgb
                import tempfile
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(model_bytes)
                    tmp.flush()
                    bst = xgb.Booster()
                    bst.load_model(tmp.name)
                payload = pickle.dumps(bst)
            else:
                payload = pickle.dumps({"model_pt": model_bytes})

            parsed_out = urlparse(self.output_dir)
            bucket_out = parsed_out.netloc
            prefix_out = parsed_out.path.lstrip('/')
            s3_key_out = os.path.join(prefix_out, f"model_{framework}.pkl")

            self.s3.put_object(Bucket=bucket_out, Key=s3_key_out, Body=payload)
            self.logger.info(f"Modelo exportado correctamente a s3://{bucket_out}/{s3_key_out}")

        except Exception as e:
            self.logger.error(f"Error en el export del modelo (S3 direct): {str(e)}", exc_info=True)

    def _load_schema_features(self) -> list:
        """Count input features and extract target from the DSL file."""
        dsl_path = self.load_params().get('spark', {}).get('preprocessing', {}).get(
            'dsl_path', '/app/repo/k3s/spark/preprocess/dsl_001.yaml'
        )
        dsl_path = os.path.join('/home/ray/', dsl_path.lstrip('/'))
        
        with open(dsl_path, 'r') as f:
            dsl = yaml.safe_load(f)
            
        final_features = dsl.get('final_features', {})
        if not isinstance(final_features, dict):
            raise ValueError("DSL final_features must be an object with keys 'features' and 'target'")

        features = final_features.get("features", [])
        
        # Override target if present in DSL
        dsl_target = final_features.get("target", [])
        if dsl_target and isinstance(dsl_target, list):
            self.target_col = dsl_target[0]
            self.logger.info(f"Target column from DSL: {self.target_col}")
        else:
            self.target_col = self.params.get('target', 'attack')
            self.logger.info(f"Target column using fallback: {self.target_col}")

        self.input_dim = len(features)
        self.logger.info(f"Input dimension calculated from DSL: {self.input_dim} features.")
        return features

    def _load_hyperparams(self, framework: str, defaults: dict) -> dict:
        """Merge YAML hyperparameter overrides with framework defaults.

        Reads from params.yaml → hyperparams → {framework} (new format) or
        params.yaml → kuberay.model.{framework}_params (legacy format).
        Validates that only known keys are passed — raises ValueError on any unknown key.
        """
        from src.schemas.model.xgboost_params import XGBOOST_ALLOWED_KEYS
        from src.schemas.model.pytorch_params import PYTORCH_ALLOWED_KEYS

        allowed = XGBOOST_ALLOWED_KEYS if framework == "xgboost" else PYTORCH_ALLOWED_KEYS

        # New format: params.yaml → hyperparams → {framework}
        yaml_overrides = self.params_full.get("hyperparams", {}).get(framework, {})
        # Legacy fallback: params.yaml → kuberay.model.{framework}_params
        if not yaml_overrides:
            yaml_overrides = self.params.get(f"{framework}_params", {})

        if yaml_overrides:
            invalid = set(yaml_overrides) - allowed
            if invalid:
                raise ValueError(
                    f"Invalid hyperparameters for {framework}: {sorted(invalid)}. "
                    f"Allowed keys: {sorted(allowed)}"
                )

        return {**defaults, **yaml_overrides}

    def _log_final_to_mlflow(
        self,
        *,
        framework: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        model=None,
        artifact_set_id: str = None,
        table_identifier: str = None,
    ) -> None:
        """Log metrics, artifacts, and register the model in MLflow Model Registry."""
        artifact_location = (
            params.get("mlflow_artifact_location", "s3://k8s-mlops-platform-bucket/mlflow-artifacts/")
        )
        registry_model_name = params.get("mlflow_registry_model_name")
        train_run_id = self.params_full.get("run_metadata", {}).get("train_run_id", "")

        try:
            result_info = log_training_run(
                framework=framework,
                params=params,
                metrics=metrics,
                artifact_location=artifact_location,
                artifact_set_id=artifact_set_id,
                table_identifier=table_identifier,
                train_run_id=train_run_id or None,
                model=model,
                registry_model_name=registry_model_name,
            )
            if result_info:
                self.logger.info(
                    "MLflow run completed: run_id=%s, model_version=%s, registry=%s",
                    result_info.get("run_id"),
                    result_info.get("model_version"),
                    result_info.get("registry_model_name"),
                )
        except Exception as e:
            self.logger.error(f"Error al loggear en MLflow: {str(e)}", exc_info=True)
            return None

        return result_info

    def _extract_model(self, result, framework: str):
        """Extract the trained model object from a Ray Train checkpoint.

        Used for logging to MLflow Model Registry with native flavors.
        """
        checkpoint = result.checkpoint
        if not checkpoint:
            self.logger.warning("No checkpoint found in training result.")
            return None

        try:
            if framework == "xgboost":
                booster = RayTrainReportCallback.get_model(checkpoint)
                # Embed feature names so mlflow.xgboost.log_model serialises them
                # into the artifact.  At serving time the adapter reads them back via
                # booster.feature_names — without this they are None and the
                # dimension-validation in adapters.py raises TypeError.
                if booster is not None and self.schema:
                    booster.feature_names = [str(f) for f in self.schema]
                    self.logger.info(
                        "Set feature_names on booster (%d features)", len(self.schema)
                    )
                return booster

            if framework == "pytorch":
                import torch
                from src.models.registry import get_model
                num_classes = int(self.params.get("num_classes", 6))
                model_type = self.params.get("model_type", "mlp")
                model = get_model(model_type, {"input_dim": self.input_dim, "num_classes": num_classes})
                with checkpoint.as_directory() as ckpt_dir:
                    state = torch.load(
                        os.path.join(ckpt_dir, "model.pt"),
                        map_location="cpu",
                    )
                    model.load_state_dict(state["model_state_dict"])
                model.eval()
                return model

            self.logger.warning(f"Unknown framework '{framework}' for model extraction.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract model from checkpoint: {e}", exc_info=True)
            return None

    # ──────────────────────────────────────────────
    # Main training entry point
    # ──────────────────────────────────────────────

    def train(self):
        # Priority: MODEL_TYPE env var > params.yaml > default
        framework = os.getenv("MODEL_TYPE") or self.params.get("framework", "xgboost")
        job_start_time = time.time()

        try:
            # Log cluster resources
            status = subprocess.run(
                ["ray", "status"], capture_output=True, text=True, check=False
            )
            stdout = status.stdout
            cpu = re.search(r"([\d.]+)/([\d.]+) CPU", stdout)
            mem = re.search(r"([\dA-Za-z.]+)/([\dA-Za-z.]+) memory", stdout)
            obj = re.search(r"([\dA-Za-z.]+)/([\dA-Za-z.]+) object_store_memory", stdout)
            cpu_used, cpu_total = cpu.groups() if cpu else ("?", "?")
            mem_used, mem_total = mem.groups() if mem else ("?", "?")
            obj_used, obj_total = obj.groups() if obj else ("?", "?")

            pretty_log = f"""
            [RAY CLUSTER RESOURCES]
            ────────────────────────────────
            CPU           : {cpu_used} / {cpu_total}
            Memory        : {mem_used} / {mem_total}
            Object Store  : {obj_used} / {obj_total}
            ────────────────────────────────
            """.strip()

            self.logger.info(pretty_log)
            self._check_minio_connection()

            # Resolve Iceberg processed table from metadata
            artifact_set_id, processed_table_full = self._get_latest_artifact()
            if not processed_table_full:
                raise ValueError(
                    "No processed table found in metadata. "
                    "Preprocessing must run first."
                )
            self.logger.info(f"Using processed table from metadata: {processed_table_full}")

            # Strip catalog prefix ("iceberg") → pyiceberg 2-part identifier
            # e.g. "iceberg.processed.network_traffic_raw_20260223_050000"
            #    → "processed.network_traffic_raw_20260223_050000"
            table_identifier = ".".join(processed_table_full.split(".")[1:])
            catalog_config = dict(self.catalog_kwargs)

            self.logger.info(f"Starting training pipeline with Iceberg table: {table_identifier}")
            self.logger.info(f"Starting training using framework: {framework}")

            num_classes = int(self.params.get("num_classes", 2))
            input_dim = (
                self.input_dim
                if self.params.get("dsl_count_dim", True)
                else int(self.params.get("input_dim", 14))
            )
            mlflow_tracking_uri = self.params.get("mlflow_tracking_uri")
            mlflow_experiment_name = self.params.get("mlflow_experiment_name")

            # ── Resolve GPU flag early so both tuning and training honour it ──
            # Priority: USE_GPU env var (set by SkyPilot YAML) > params.yaml use_gpu.
            # This prevents params.yaml "use_gpu: false" from clobbering the
            # "USE_GPU: true" injected by ray-gpu-training.yaml at VM launch time.
            _env_gpu = os.getenv("USE_GPU", "auto").lower()
            if _env_gpu in ("true", "1", "yes"):
                use_gpu_flag = True
            elif _env_gpu in ("false", "0", "no"):
                use_gpu_flag = False
            else:  # "auto" — fall back to params.yaml
                use_gpu_flag = self.params.get("use_gpu", False)
            os.environ["USE_GPU"] = "true" if use_gpu_flag else "false"
            self.logger.info(
                "GPU mode: %s (USE_GPU=%s, source=%s)",
                "enabled" if use_gpu_flag else "disabled",
                os.environ["USE_GPU"],
                "env" if _env_gpu != "auto" else "params.yaml",
            )

            # ── Tuning phase (optional) ──
            best_params = None
            if self.params.get('tune', False):
                self.logger.info("Starting hyperparameter tuning with Iceberg row filtering...")

                tuner = get_tuner(framework)
                prom_tune_cb = PrometheusTuneCallback(
                    framework=framework,
                    metric_name=tuner.tune_metric,
                    mode=tuner.tune_mode,
                )

                best_config = tuner.tune_model(
                    table_identifier=table_identifier,
                    catalog_config=catalog_config,
                    split_ranges=self.params_full.get('splits', {}),
                    target=self.target_col,
                    feature_columns=self.schema,
                    sample_fraction=self.params.get('sample_fraction_for_tuning'),
                    seed=int(self.params.get('seed', 42)),
                    storage_path=self.output_dir,
                    name=self.params.get('name', framework) + "_tune",
                    input_dim=input_dim,
                    num_classes=num_classes,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_experiment_name=mlflow_experiment_name,
                    extra_callbacks=[prom_tune_cb],
                    number_of_trials=self.params_full.get('execution', {}).get('tuning', {}).get('number_of_trials'),
                )

                best_params = best_config.get(tuner.params_key)
                self.logger.info(f"Best hyperparameters found: {best_params}")

            # ── Load datasets ──
            self.logger.info("Loading final datasets from Iceberg...")
            train_ds = self._load_data('train', table_identifier)
            val_ds = self._load_data('val', table_identifier)
            test_ds = self._load_data('test', table_identifier)

            # Export dataset sizes for Grafana (best-effort)
            try:
                if os.getenv('EXPORT_DATASET_COUNTS', '1').lower() in ('1', 'true', 'yes'):
                    TRAIN_SPLIT_ROWS.labels(framework=framework, split='train').set(float(train_ds.count()))
                    TRAIN_SPLIT_ROWS.labels(framework=framework, split='val').set(float(val_ds.count()))
                    TRAIN_SPLIT_ROWS.labels(framework=framework, split='test').set(float(test_ds.count()))
            except Exception as e:
                self.logger.warning(f"Could not export dataset counts: {e}")

            # ── Resolve effective params ──
            # Base defaults come from the Python schema (XGBOOST_PARAMS / PYTORCH_PARAMS).
            # YAML overrides from params.yaml → hyperparams → {framework} are merged on top
            # and validated against the allowed-keys allowlist before training starts.
            defaults, params_key = _FRAMEWORK_DEFAULTS[framework]
            base = self._load_hyperparams(framework, dict(defaults))

            if best_params is None:
                effective_params = base
            else:
                effective_params = {**base, **best_params}
                # Ensure long-training keys are present (not in tune search spaces)
                if framework == "xgboost" and "num_boost_round" not in effective_params:
                    effective_params["num_boost_round"] = base["num_boost_round"]
                if framework == "pytorch" and "max_epochs" not in effective_params:
                    effective_params["max_epochs"] = base["max_epochs"]

            # ── Train ──
            task_type = self.params.get("task_type", "classification")
            model_type = self.params.get("model_type", "mlp")
            trainer = get_trainer(framework)
            result, final_metrics = trainer.train(
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
                storage_path=self.output_dir,
                name=self.params.get('name', framework),
                target=self.target_col,
                feature_columns=self.schema,
                input_dim=input_dim,
                num_classes=num_classes,
                params=effective_params,
                task_type=task_type,
                model_type=model_type,
            )
            self.logger.info("Training completed successfully.")

            # ── Prometheus final metrics ──
            if final_metrics:
                try:
                    export_final_metrics(framework=framework, metrics=final_metrics)
                    self.logger.info("Final metrics exported to Prometheus")
                except Exception as e:
                    self.logger.warning(f"Could not export final metrics to Prometheus: {e}")

            # ── Save model to S3 (backward compat with serving) ──
            self._save_model(result, framework)

            # ── MLflow logging ──
            model_obj = self._extract_model(result, framework)

            mlflow_payload = {
                **self.params,
                "name": self.params.get('name', framework),
                params_key: effective_params,
            }

            mc_time_sec = final_metrics.get("multiclass_metrics_time_sec")
            if mc_time_sec is not None:
                self.logger.info(
                    "%s multiclass metrics time = %.2f s",
                    framework,
                    float(mc_time_sec),
                )

            mlflow_result = self._log_final_to_mlflow(
                framework=framework,
                params=mlflow_payload,
                metrics=final_metrics,
                model=model_obj,
                artifact_set_id=artifact_set_id,
                table_identifier=table_identifier,
            )

            # ── Cost tracking ────────────────────────────────────────────────
            gpu_price = float(os.getenv("GPU_PRICE_PER_HOUR", "0") or "0")
            is_spot = os.getenv("USE_SPOT", "false").lower() == "true"
            if gpu_price > 0 and mlflow_result and mlflow_result.get("run_id"):
                elapsed_hours = (time.time() - job_start_time) / 3600
                estimated_cost = round(gpu_price * elapsed_hours, 4)
                try:
                    import mlflow as _mlflow
                    tracking_uri = self.params.get("mlflow_tracking_uri")
                    if tracking_uri:
                        _mlflow.set_tracking_uri(tracking_uri)
                    with _mlflow.start_run(run_id=mlflow_result["run_id"]):
                        _mlflow.set_tag("estimated_cost_usd", estimated_cost)
                        _mlflow.set_tag("instance_type", "spot" if is_spot else "on_demand")
                        _mlflow.set_tag("gpu_price_per_hour", gpu_price)
                    self.logger.info(
                        "Cost tags logged: $%.4f  (%.2fh × $%.3f/h, %s)",
                        estimated_cost, elapsed_hours, gpu_price,
                        "spot" if is_spot else "on_demand",
                    )
                except Exception as ce:
                    self.logger.warning(f"Could not log cost tags to MLflow: {ce}")

            # Prometheus scrape grace period
            grace = int(os.getenv('PROMETHEUS_GRACE_SECONDS', '5'))
            if grace > 0:
                self.logger.info(f"Sleeping {grace}s for Prometheus scrape grace period...")
                time.sleep(grace)

            return result

        except Exception as e:
            TRAIN_FAILURES.labels(framework=framework, error_type=type(e).__name__).inc()
            self.logger.error(f'Training job failed: {str(e)}', exc_info=True)
            raise


def _resolve_params_path() -> str:
    """Resolve local params file path.

    Priority:
      1. PARAMS_PATH env var (explicit local path — manual runs, testing)
      2. PARAMS_S3_PATH env var — download params_training.yaml from S3 to a temp file
         (Airflow-triggered runs: the DAG injects the per-run S3 URI here)
      3. Static git-synced reference file (fallback for legacy / local dev)
    """
    explicit = os.getenv("PARAMS_PATH")
    if explicit:
        return explicit

    s3_uri = os.getenv("PARAMS_S3_PATH")
    if s3_uri:
        import tempfile
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-2"),
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        s3.download_file(bucket, key, tmp.name)
        print(f"Downloaded params from {s3_uri} → {tmp.name}")
        return tmp.name

    return "/home/ray/app/repo/k3s/params.yaml"


def main():
    if not ray.is_initialized():
        # Port 6379: the Ray cluster started by ~/sky_templates/ray/start_cluster.
        # "auto" would connect to SkyPilot's internal Ray on port 6380 instead.
        ray.init(address="localhost:6379", ignore_reinit_error=True)

    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    output_dir = os.getenv("OUTPUT_DIR", "s3://k8s-mlops-platform-bucket/v1/models")

    model = KubeRayTraining(
        params_path=_resolve_params_path(),
        output_dir=output_dir
    )
    model.train()


if __name__ == "__main__":
    main()
