from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from src.dsl import Pipeline, PipelineModel
import tempfile
from src.schemas.spark.schema_registry import get_schema
# ===== PROMETHEUS METRICS =====
from prometheus_client import start_http_server
from src.prometheus import (
    PREPROCESS_RECORDS,
    PREPROCESS_BATCHES,
    PREPROCESS_RECORDS_BY_CLASS,
    PREPROCESS_RECORDS_LAST_BATCH,
    PREPROCESS_BATCH_LATENCY,
    PREPROCESS_BATCH_LATENCY_LAST,
    PREPROCESS_ERRORS,
    PREPROCESS_CURRENT_DATASET
)

class SparkPreprocessing(BaseUtils):
    def __init__(self, params_path: str, data_dir: str, output_dir: str, artifacts_dir: str):
        logger = create_logger('SparkPreprocessing', 'spark_preprocessing.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['spark']
        self.data_dir = data_dir
        self.output_dir = output_dir
        self._load_artifacts_config()
        self.schema = get_schema(self.params.get('schema').get('schema_full'))
        self.s3 = None
        self.spark = self._create_spark_session()
        self.pipeline_path = self.params['preprocessing'].get('dsl_path')
        self.pipeline = None
        
        # Start Prometheus metrics server
        self._start_prometheus_server()
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server on port 8001."""
        port = int(os.getenv('PROMETHEUS_PORT', 8001))
        try:
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")

            # Pre-create expected labelsets so Grafana panels show train/val/test even if a split has 0 rows.
            for split in ("train", "val", "test"):
                PREPROCESS_RECORDS.labels(dataset=split).inc(0)
                PREPROCESS_BATCHES.labels(dataset=split).inc(0)
                PREPROCESS_BATCH_LATENCY_LAST.labels(dataset=split).set(0)

                # Pre-create class labelsets (default 2 classes: 0..1)
                num_classes = self.params.get('num_classes', 2)
                for c in range(max(num_classes, 1)):
                    PREPROCESS_RECORDS_LAST_BATCH.labels(dataset=split, **{'class': str(c)}).set(0)
        except Exception as e:
            self.logger.warning(f"Could not start Prometheus server: {e}")

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

    @staticmethod
    def _to_s3a_path(path: str) -> str:
        if path.startswith('s3a://'):
            return path
        if path.startswith('s3://'):
            return 's3a://' + path[len('s3://'):]
        return path
        
    def _create_spark_session(self):
        try:
            self._check_minio_connection()
            self.logger.info("Creating SparkSession with S3A (MinIO) support")
            spark = (
            SparkSession.builder
            .appName(self.params['app_name'])
            .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
            
            # 1. Especificar el proveedor de credenciales (evita confusiones internas de Hadoop)
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .config("spark.hadoop.fs.s3a.path.style.access", "false")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            
            # 2. Rendimiento: Esto ayuda mucho con Parquets grandes en S3
            .config("spark.hadoop.fs.s3a.experimental.fadvise", "random")
            .config("spark.sql.files.maxPartitionBytes", self.params.get('read_batch_size', 256) * 1024 * 1024)
            
            # 3. Magic Committer (Ya lo tenías, ¡excelente!)
            .config("spark.hadoop.fs.s3a.committer.name", "magic")
            .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
            .getOrCreate()
        )
        except Exception as e:
            self.logger.error('Cannot create Spark session: %s', str(e))
            raise

        return spark

    def load_data(self, file_path: str):
        try:

            spark = self.spark
            self.logger.info(f"Loading data from {file_path}")
            df = spark.read.schema(self.schema).parquet(file_path) #Aqui tambien se valida el schema
            self.logger.info(
                f"Data loaded successfully | partitions: {df.rdd.getNumPartitions()}"
            )
            return df

        except Exception as e:
            self.logger.error("Failed loading data", exc_info=True)
            raise

    def preprocess(self, dataset: str = 'train'):
        """
        Unified preprocessing entry with Prometheus metrics.
        """
        dataset_idx = {'train': 1, 'val': 2, 'test': 3}.get(dataset, 0)
        PREPROCESS_CURRENT_DATASET.set(dataset_idx)
        
        start_time = time.time()
        try:
            self.logger.info(f"Starting preprocessing for {dataset} dataset")
            if dataset not in ['train', 'val', 'test']:
                raise ValueError("dataset must be one of ['train', 'val', 'test']")

            self.logger.info(f"Loading feature pipeline from: {self.pipeline_path}")

            if dataset == 'train':
                df = self.load_data(os.path.join(self.data_dir, f'{dataset}/'))
                base_pipeline = Pipeline.from_yaml(self.pipeline_path)
                self.pipeline = base_pipeline.fit(df)
                self._save_pipeline_artifact()
            else:
                df = self.load_data(os.path.join(self.data_dir, f'{dataset}/'))
                if self.pipeline is None:
                    self.load_pipeline_artifact()

            df_processed = self.pipeline.transform(df)
            df_out = self.pipeline.select_features(df_processed)

            # ===== PROMETHEUS METRICS =====
            record_count = df_out.count()
            PREPROCESS_RECORDS.labels(dataset=dataset).inc(record_count)
            # NOTE: This counts how many times the dataset split ran (not Spark partitions).
            PREPROCESS_BATCHES.labels(dataset=dataset).inc(1)

            # Reset last-batch gauges to 0 for stable pie chart labels (0..num_classes-1)
            num_classes = self.params.get('num_classes', 2)

            for c in range(max(num_classes, 1)):
                PREPROCESS_RECORDS_LAST_BATCH.labels(dataset=dataset, **{'class': str(c)}).set(0)
            
            # Per-class distribution (if 'attack' column exists)
            target_col = self.params.get('target', 'attack')
            if target_col in df_out.columns:
                class_counts = df_out.groupBy(target_col).count().collect()
                for row in class_counts:
                    cls = str(row[target_col])
                    cnt = row['count']
                    PREPROCESS_RECORDS_BY_CLASS.labels(dataset=dataset, **{'class': cls}).inc(cnt)
                    PREPROCESS_RECORDS_LAST_BATCH.labels(dataset=dataset, **{'class': cls}).set(cnt)
            
            elapsed = time.time() - start_time
            PREPROCESS_BATCH_LATENCY.labels(dataset=dataset).observe(elapsed)
            PREPROCESS_BATCH_LATENCY_LAST.labels(dataset=dataset).set(elapsed)

            self.logger.info(f"Preprocessing completed for {dataset} dataset. Writing output.")
            self.write_data(df_out, os.path.join(self.output_dir, f'{dataset}/'))
            self.logger.info(f"Output written in S3 for {dataset} dataset.")
            
            # Reset progress to idle when done
            PREPROCESS_CURRENT_DATASET.set(0)
        except Exception as e:
            PREPROCESS_ERRORS.labels(dataset=dataset, error_type=type(e).__name__).inc()
            PREPROCESS_CURRENT_DATASET.set(0)
            self.logger.error('Preprocess failed to complete: %s', str(e), exc_info=True)
            raise

    def _load_artifacts_config(self):
        try:
            """Carga configuración de artifacts desde params."""
            artifacts_config = self.params.get('preprocessing').get('artifacts')
            
            # Configuración de artifacts
            self.artifacts_source = artifacts_config.get('source', 's3')
            self.artifacts_bucket = artifacts_config.get('bucket', 'k8s-mlops-platform-bucket')
            self.artifacts_key = artifacts_config.get('key', 'v1/artifacts')
            self.artifacts_list = artifacts_config.get('archives', ['config.json', 'stages.json'])
            self.logger.info(f"✅ Artifacts configured: archives={self.artifacts_list} from {self.artifacts_source}")
        except Exception as e:
            self.logger.error('Failed loading artifacts config: %s', str(e), exc_info=True)
            raise

    def _save_pipeline_artifact(self):
        """Guardar PipelineModel usando su método save() y subir los artefactos a S3."""
        try:

            # Crear un directorio temporal para que PipelineModel.save lo escriba ahí
            tmpdir = tempfile.mkdtemp(prefix="pipeline_model")
            # Usar el método save() del pipeline (espera una ruta local)
            self.pipeline.save(tmpdir)

            for i in self.artifacts_list:
                self.s3.upload_file(
                    os.path.join(tmpdir, i),
                    Bucket=self.artifacts_bucket,
                    Key=os.path.join(self.artifacts_key, i)
                )

            self.logger.info(f'PipelineModel saved to {tmpdir} and uploaded to S3 {self.artifacts_bucket}/{self.artifacts_key}')
        except Exception:
            self.logger.error('Failed saving pipeline model to S3', exc_info=True)
            raise

    def load_pipeline_artifact(self):
        """Carga el PipelineModel desde local; si no existe, lo descarga de S3."""
        try:

            tmpdir = os.path.join("/tmp", "pipeline_model")

            # 1️⃣ Si existe en local, cargarlo
            if os.path.exists(os.path.join(tmpdir, self.artifacts_list[0])):
                self.logger.info("Loading pipeline artifact from local cache")
                self.pipeline = PipelineModel.load(tmpdir)
                return self.pipeline

            # 2️⃣ Si no existe, descargar desde S3
            self.logger.info("Local pipeline not found. Downloading from S3")

            if self.s3 is None:
                self._check_minio_connection()

            os.makedirs(tmpdir, exist_ok=True)

            for i in self.artifacts_list:
                self.s3.download_file(
                    self.artifacts_bucket,
                    os.path.join(self.artifacts_key, i),
                    os.path.join(tmpdir, i)
                )

            self.pipeline = PipelineModel.load(tmpdir)
            self.logger.info("Pipeline artifact downloaded from S3 and loaded")
        except Exception:
            self.logger.error("Failed loading pipeline artifact", exc_info=True)
            raise
    
    def write_data(self, df, output_path: str):
        """
        Escribe DataFrame en parquet en S3 con particiones seguras según batch_size.
        """
        try:
            batch_size = self.params.get('write_batch_size', 100000)
            self.logger.info(
                f"Writing data | write_batch_size={batch_size}"
            )

            df.write \
              .mode("overwrite") \
              .option("maxRecordsPerFile", batch_size) \
              .parquet(output_path)

            self.logger.info(f"Data written successfully to {output_path}")

        except Exception as e:
            self.logger.error("Failed writing data", exc_info=True)
            raise


def main():     
    data_dir = "s3a://k8s-mlops-platform-bucket/v1/raw/" #Para Spark
    output_dir = "s3a://k8s-mlops-platform-bucket/v1/processed/" #Para Spark
    artifacts_dir = "s3a://k8s-mlops-platform-bucket/v1/artifacts/"

    preprocessing = SparkPreprocessing(
        params_path="/app/repo/k3s/params.yaml",
        data_dir=data_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir
    )

    # 1) Preprocess TRAIN and fit scaler
    preprocessing.preprocess('train')

    # 2) Preprocess VAL using train-fitted transforms
    preprocessing.preprocess('val')

    # 3) Preprocess TEST using train-fitted transforms
    preprocessing.preprocess('test')

    preprocessing.logger.info("Spark preprocessing completed successfully.")

    # Optional: keep the driver alive briefly so Prometheus can scrape final metrics.
    try:
        grace = int(os.getenv('PROMETHEUS_GRACE_SECONDS', '0'))
    except Exception:
        grace = 0
    if grace > 0:
        preprocessing.logger.info(f"Sleeping {grace}s for Prometheus scrape grace period...")
        time.sleep(grace)

if __name__ == "__main__":
    main()
