from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import ray
import re
import os
import importlib
import pickle
import subprocess
import yaml
import time
import tempfile
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import GreaterThanOrEqual, LessThanOrEqual, And
from urllib.parse import urlparse
from typing import Dict, Any
from ray.train.xgboost import RayTrainReportCallback

from src.schemas.model.pytorch_params import PYTORCH_PARAMS
from src.schemas.model.xgboost_params import XGBOOST_PARAMS
from src.pipeline.utils.mlflow_utils import log_training_run
from src.schemas.spark.schema_registry import SchemaRegistry

# ===== PROMETHEUS METRICS =====
# Keep metrics in an importable module so Ray Tune can pickle them.
from prometheus_client import start_http_server
from src.prometheus import (
    TRAIN_FAILURES,
    TRAIN_SPLIT_ROWS,
    PrometheusTuneCallback,
    export_final_metrics,
)
# Training metrics (accuracy, loss, etc.) are now exported directly from worker
# training loops in pytorch_utils.py/xgboost_utils.py, not via callbacks.

class KubeRayTraining(BaseUtils):
    def __init__(self, params_path: str, output_dir: str):
        logger = create_logger('KubeRayTraining', 'kuberay_training.log')
        super().__init__(logger, params_path)
        self.params_full = self.load_params()
        self.params = self.params_full['kuberay']['model']
        self.schema = self._load_schema_features()
        self.output_dir = output_dir
        
        # Iceberg setup
        self.catalog = self._get_iceberg_catalog()
        
        # Start Prometheus metrics server
        self._start_prometheus_server()

    def _get_iceberg_catalog(self):
        """Initializes PyIceberg catalog using configuration from Spark/S3."""
        warehouse = self.params_full.get('spark', {}).get('iceberg', {}).get(
            'warehouse',
            's3a://k8s-mlops-platform-bucket/warehouse'
        ).replace('s3a://', 's3://')

        self.logger.info(f"Initializing Iceberg catalog with warehouse: {warehouse}")
        return load_catalog("iceberg", **{
            "type": "glue",
            "warehouse": warehouse,
            "s3.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "s3.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "s3.region": os.environ.get("AWS_REGION", "us-east-2"),
        })

    def _get_latest_artifact_set_id(self):
        """Retrieves the latest artifact_set_id from the metadata table."""
        try:
            meta_cfg = self.params_full.get('iceberg_tables', {}).get('metadata', {})
            # Identifier for pyiceberg (removing the catalog prefix)
            identifier = f"{meta_cfg.get('namespace', 'metadata')}.{meta_cfg.get('table', 'preprocessing_artifacts')}"
            
            table = self.catalog.load_table(identifier)
            df = table.scan().to_pandas()
            if df.empty:
                return None
            
            latest = df.sort_values('created_at', ascending=False).iloc[0]
            self.logger.info(f"Latest artifact_set_id found in metadata: {latest['artifact_set_id']}")
            return latest['artifact_set_id']
        except Exception as e:
            self.logger.warning(f"Could not find latest artifact_set_id from metadata: {e}")
            return None
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server on port 8002."""
        port = int(os.getenv('PROMETHEUS_PORT', 8002))
        print(f"[main.py] Attempting to start Prometheus server on port {port}...")
        try:
            start_http_server(port)
            self.logger.info(f"✓ Prometheus metrics server started on port {port}")
            print(f"[main.py] ✓ Prometheus HTTP server successfully started on port {port}")
        except OSError as e:
            if "Address already in use" in str(e):
                self.logger.info(f"Prometheus server already running on port {port}")
                print(f"[main.py] Prometheus server already running on port {port} (OK)")
            else:
                self.logger.warning(f"Could not start Prometheus server: {e}")
                print(f"[main.py] ERROR starting Prometheus server: {e}")
        except Exception as e:
            self.logger.warning(f"Could not start Prometheus server: {e}")
            print(f"[main.py] ERROR starting Prometheus server: {e}")

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
            start_ts = split_cfg.get('start')
            end_ts = split_cfg.get('end')

            row_filter = None
            if start_ts and end_ts:
                self.logger.info(f"Loading {split} split from {table_name} ({start_ts} to {end_ts})")
                row_filter = And(
                    GreaterThanOrEqual("timestamp", start_ts),
                    LessThanOrEqual("timestamp", end_ts)
                )
            else:
                self.logger.warning(f"No valid split configuration for {split}. Loading all data.")

            ds = ray.data.read_iceberg(
                table_name,
                catalog=self.catalog,
                row_filter=row_filter
            )
            
            # Senior validation before heavy processing
            self._validate_schema(ds)

            # Senior Optimization: Materialize data in the Object Store once.
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
                self.schema = SchemaRegistry.get_schema('schema_preprocessed')  # Ensure schema is registered

            expected = {
                self.params.get('target', 'attack'),
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
        """
        Extrae el mejor modelo del resultado y lo guarda en S3 como un archivo .pkl.
        Usa boto3 directo para evitar archivos temporales y errores de checksum.
        """
        self.logger.info(f"Exportando modelo final de {framework} a S3...")
        try:
            checkpoint = result.checkpoint
            if not checkpoint:
                self.logger.warning("No se encontró un checkpoint válido en el resultado.")
                return

            # 1. Parsear el path de S3 (manejando si tiene o no el scheme s3://)
            ckpt_path = checkpoint.path
            raw_path = ckpt_path[5:] if ckpt_path.startswith("s3://") else ckpt_path
            bucket_in, prefix_in = raw_path.split("/", 1)

            # 2. Definir archivo de origen según framework
            target_file = "model.pt" if framework == "pytorch" else "model.ubj"
            key_in = f"{prefix_in.rstrip('/')}/{target_file}"

            self.logger.info(f"Descargando {target_file} desde s3://{bucket_in}/{key_in}")
            response = self.s3.get_object(Bucket=bucket_in, Key=key_in)
            model_bytes = response['Body'].read()

            # 3. Preparar el payload final para el archivo .pkl
            # Mantenemos la estructura original del usuario para no romper la inferencia
            if framework == "xgboost":
                # RayTrainReportCallback.get_model(checkpoint) devuelve el objeto Booster
                # Aquí lo simulamos cargando los bytes si fuera necesario, o simplemente
                # guardamos los bytes dentro del pickle si así lo espera el consumidor.
                # Para ser 100% fieles al código original que usaba pickle.dump(model):
                import xgboost as xgb
                import tempfile
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(model_bytes)
                    tmp.flush()
                    bst = xgb.Booster()
                    bst.load_model(tmp.name)
                payload = pickle.dumps(bst)
            else:
                # PyTorch: dict con la key "model_pt"
                payload = pickle.dumps({"model_pt": model_bytes})

            # 4. Subir al destino final
            parsed_out = urlparse(self.output_dir)
            bucket_out = parsed_out.netloc
            prefix_out = parsed_out.path.lstrip('/')
            s3_key_out = os.path.join(prefix_out, f"model_{framework}.pkl")

            self.s3.put_object(Bucket=bucket_out, Key=s3_key_out, Body=payload)
            self.logger.info(f"Modelo exportado correctamente a s3://{bucket_out}/{s3_key_out}")
                
        except Exception as e:
            self.logger.error(f"Error en el export del modelo (S3 direct): {str(e)}", exc_info=True)
    
    def _load_schema_features(self) -> list:
        """Cuenta dinámicamente las columnas de entrada (features) en el dataset."""
        dsl_path = self.load_params().get('spark', {}).get('preprocessing',{}).get('dsl_path', '/app/repo/k3s/spark/preprocess/dsl_001.yaml')
        dsl_path = os.path.join('/home/ray/', dsl_path.lstrip('/'))
        with open(dsl_path, 'r') as f:
            dsl = yaml.safe_load(f)
        final_features = dsl['pipeline']['final_features']

        categorical = final_features.get("categorical", [])
        numerical = final_features.get("numerical", [])

        self.input_dim = len(categorical) + len(numerical)
        self.logger.info(f"Input dimension calculated from DSL: {self.input_dim} features.")
        return categorical + numerical

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

        try:
            result_info = log_training_run(
                framework=framework,
                params=params,
                metrics=metrics,
                artifact_location=artifact_location,
                artifact_set_id=artifact_set_id,
                table_identifier=table_identifier,
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

    def _extract_model(self, result, framework: str):
        """Extract the trained model object from a Ray Train checkpoint.

        Used for logging to MLflow Model Registry with native flavors
        (mlflow.xgboost / mlflow.pytorch).
        """
        checkpoint = result.checkpoint
        if not checkpoint:
            self.logger.warning("No checkpoint found in training result.")
            return None

        try:
            if framework == "xgboost":
                booster = RayTrainReportCallback.get_model(checkpoint)
                return booster
            elif framework == "pytorch":
                import torch
                from src.models.pytorch import NeuralNetwork
                num_classes = int(self.params.get("num_classes", 6))
                model = NeuralNetwork(input_dim=self.input_dim, num_classes=num_classes)
                with checkpoint.as_directory() as ckpt_dir:
                    state = torch.load(
                        os.path.join(ckpt_dir, "model.pt"),
                        map_location="cpu",
                    )
                    model.load_state_dict(state["model_state_dict"])
                model.eval()
                return model
            else:
                self.logger.warning(f"Unknown framework '{framework}' for model extraction.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to extract model from checkpoint: {e}", exc_info=True)
            return None

    def _stratified_sample(self, ds, target_col, fraction):
        """
        Realiza un muestreo estratificado distribuido usando Ray Data.
        Asegura que cada clase tenga al menos un número mínimo de muestras para no perder ninguna.
        """
        self.logger.info(f"Realizando muestreo estratificado ({fraction*100}%) sobre la columna '{target_col}'...")

        def sample_group(df):
            # Calculamos cuántas muestras representa la fracción
            n = int(len(df) * fraction)
            # Aseguramos al menos 5 muestras (o el total si el grupo es más pequeño) 
            # para no perder clases minoritarias en el tuning.
            n_final = max(min(len(df), 5), n)
            return df.sample(n=n_final, random_state=self.params.get('seed', 42))

        # map_groups permite aplicar operaciones sobre cada grupo de forma distribuida
        return ds.groupby(target_col).map_groups(sample_group)
    
    def train(self):
        try:
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
            module = importlib.import_module(f"{'src.pipeline.train.' + self.params.get('framework', 'xgboost')}")

            framework = self.params.get("framework", "xgboost")

            artifact_set_id = self._get_latest_artifact_set_id()

            if not artifact_set_id:
                raise ValueError("ARTIFACT_SET_ID not found and no metadata records available. Preprocessing must run first.")

            self.logger.info(f"Using latest artifact_set_id from metadata: {artifact_set_id}")

            table_identifier = f"processed.{artifact_set_id}"
            self.logger.info(f"Starting training pipeline with Iceberg table: {table_identifier}")

            warehouse = self.params_full.get('spark', {}).get('iceberg', {}).get(
                'warehouse',
                's3a://k8s-mlops-platform-bucket/warehouse'
            ).replace('s3a://', 's3://')
            
            catalog_config = {
                "type": "glue",
                "warehouse": warehouse,
                "s3.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID"),
                "s3.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                "s3.region": os.environ.get("AWS_REGION", "us-east-2"),
            }

            self.logger.info(f"Starting training using framework: {framework}")
            best_params = None
            num_classes = int(self.params.get("num_classes", 2))
            input_dim = self.input_dim if self.params.get("dsl_count_dim", True) else int(self.params.get("input_dim", 14))
            mlflow_tracking_uri = self.params.get("mlflow_tracking_uri")
            mlflow_experiment_name = self.params.get("mlflow_experiment_name")
            
            if self.params.get('tune', False):
                self.logger.info("Starting hyperparameter tuning with Iceberg row filtering...")

                sample_frac = self.params.get('sample_fraction_for_tuning')
                tuner = importlib.import_module('src.pipeline.tuning.'+ framework)

                if framework == 'xgboost':
                    tune_metric = 'validation-mlogloss'
                elif framework == 'pytorch':
                    tune_metric = 'val_loss'
                else:
                    tune_metric = 'accuracy'

                tune_mode = 'min'
                prom_tune_cb = PrometheusTuneCallback(framework=framework, metric_name=tune_metric, mode=tune_mode)

                best_config = tuner.tune_model(
                    table_identifier=table_identifier,
                    catalog_config=catalog_config,
                    split_ranges=self.params_full.get('splits', {}),
                    target=self.params['target'],
                    feature_columns=self.schema,
                    sample_fraction=sample_frac,
                    seed=int(self.params.get('seed', 42)),
                    storage_path=self.output_dir,
                    name=self.params.get('name', framework) + "_tune",
                    input_dim=input_dim,
                    num_classes=num_classes,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_experiment_name=mlflow_experiment_name,
                    extra_callbacks=[prom_tune_cb],
                )

                best_params = best_config.get(framework + "_params")
                self.logger.info(f"Best hyperparameters found: {best_params}")

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

            train_kwargs = {
                "train_dataset": train_ds,
                "val_dataset": val_ds,
                "test_dataset": test_ds,
                "storage_path": self.output_dir,
                "name": self.params.get('name', framework),
                "target": self.params.get('target', 'attack'),
                "feature_columns": self.schema,
                "input_dim": input_dim,
                "num_classes": num_classes,
            }

            # XGBoost specific tuned params
            if framework == "xgboost":
                if best_params is None:
                    best_params = dict(XGBOOST_PARAMS)
                # Siempre asegurar que tenga num_boost_round (no está en SEARCH_SPACE)
                best_params["num_boost_round"] = XGBOOST_PARAMS["num_boost_round"]
                train_kwargs["xgboost_params"] = best_params

            # PyTorch specific tuned params
            if framework == "pytorch":
                if best_params is None:
                    best_params = dict(PYTORCH_PARAMS)
                else:
                    # Si viene del tuning, asegurar que tenga max_epochs (no está en SEARCH_SPACE)
                    if "max_epochs" not in best_params:
                        best_params["max_epochs"] = PYTORCH_PARAMS["max_epochs"]
                train_kwargs["pytorch_params"] = best_params

            # Metrics are exported directly from worker training loop (pytorch_utils.py)
            train_out = module.train(**train_kwargs)
            if isinstance(train_out, tuple) and len(train_out) == 2:
                result, final_metrics = train_out
            else:
                result, final_metrics = train_out, {}
            self.logger.info("Training completed successfully.")
            
            # Export final metrics snapshot for gauge panels (best-effort)
            if final_metrics:
                try:
                    export_final_metrics(framework=framework, metrics=final_metrics)
                    self.logger.info("Final metrics exported to Prometheus")
                except Exception as e:
                    self.logger.warning(f"Could not export final metrics to Prometheus: {e}")
            # if getattr(prom_train_cb, '_last_step', 0) <= 0:
            #     ...epoch counter export...
            # if 'val_accuracy' in final_metrics:
            #     TRAIN_ACCURACY.labels(framework=framework, split='val').set(...)
            # etc.

            # Guardar el modelo en S3 al finalizar (backward compat con serving)
            self._save_model(result, framework)

            # Extract model from checkpoint for MLflow Model Registry
            model_obj = self._extract_model(result, framework)

            # Log en MLflow con Model Registry y trazabilidad completa
            mlflow_payload = {
                **self.params,
                "name": train_kwargs.get("name", framework),
                "xgboost_params": train_kwargs.get("xgboost_params"),
                "pytorch_params": train_kwargs.get("pytorch_params"),
            }

            mc_time_sec = final_metrics.get("multiclass_metrics_time_sec")
            if mc_time_sec is not None:
                self.logger.info(
                    "%s multiclass metrics time = %.2f s",
                    framework,
                    float(mc_time_sec),
                )

            self._log_final_to_mlflow(
                framework=framework,
                params=mlflow_payload,
                metrics=final_metrics,
                model=model_obj,
                artifact_set_id=artifact_set_id,
                table_identifier=table_identifier,
            )

            # Allow Prometheus a small window to scrape the final metric values
            # before the RayJob tears down the head pod.
            grace = int(os.getenv('PROMETHEUS_GRACE_SECONDS', '5'))
            if grace > 0:
                self.logger.info(f"Sleeping {grace}s for Prometheus scrape grace period...")
                time.sleep(grace)

            return result
        except Exception as e:
            TRAIN_FAILURES.labels(framework=framework, error_type=type(e).__name__).inc()
            self.logger.error(f'Training job failed: {str(e)}', exc_info=True)
            raise

def main():
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # For Iceberg, we don't strictly need a data_dir path as the catalog knows where data is.
    output_dir = os.getenv("OUTPUT_DIR", "s3://k8s-mlops-platform-bucket/v1/models")

    model = KubeRayTraining(
        params_path=os.getenv("PARAMS_PATH", "/home/ray/app/repo/k3s/params.yaml"),
        output_dir=output_dir
    )
    model.train()

if __name__ == "__main__":
    main()