from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import os
import time
from pyspark.sql import SparkSession
import importlib
from pyspark.sql.types import StructType
import json

# ===== PROMETHEUS METRICS =====
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Preprocessing metrics
PREPROCESS_RECORDS = Counter('preprocess_records_total', 'Total records preprocessed', ['dataset'])
PREPROCESS_BATCHES = Counter('preprocess_batches_total', 'Total batches preprocessed', ['dataset'])
PREPROCESS_RECORDS_BY_CLASS = Counter('preprocess_records_by_class_total', 'Records per class', ['dataset', 'class'])
PREPROCESS_RECORDS_LAST_BATCH = Gauge('preprocess_records_last_batch', 'Records in last batch by class', ['dataset', 'class'])
PREPROCESS_BATCH_LATENCY = Histogram('preprocess_batch_latency_seconds', 'Batch preprocessing latency', ['dataset'], buckets=[1, 5, 10, 30, 60, 120, 300])
PREPROCESS_ERRORS = Counter('preprocess_errors_total', 'Total preprocessing errors', ['dataset', 'error_type'])
PREPROCESS_CURRENT_DATASET = Gauge('preprocess_current_dataset_progress', 'Current dataset being processed (0=idle, 1=train, 2=val, 3=test)')


class SparkPreprocessing(BaseUtils):
    def __init__(self, schema: StructType,  params_path: str, data_dir: str, output_dir: str, artifacts_dir: str):
        logger = create_logger('SparkPreprocessing', 'spark_preprocessing.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['spark']
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.artifacts_dir = artifacts_dir
        self.schema = schema
        self.s3 = None
        self.spark = self._create_spark_session()
        self.scaler = None
        
        # Start Prometheus metrics server
        self._start_prometheus_server()
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server on port 8001."""
        port = int(os.getenv('PROMETHEUS_PORT', 8001))
        try:
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")
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

            pipeline_module = self.params['preprocessing']['module']
            self.logger.info(f"Loading feature pipeline: {pipeline_module}")
            module = importlib.import_module(pipeline_module)

            if dataset == 'train':
                df = self.load_data(os.path.join(self.data_dir, f'{dataset}/'))
                df_out, pipeline_model = module.preprocess_spark(df, model=self.scaler, train=True)
                self.scaler = pipeline_model
                self._save_scaler_artifact()
            else:
                df = self.load_data(os.path.join(self.data_dir, f'{dataset}/'))
                if self.scaler is None:
                    self.scaler = self.load_scaler_artifact()
                df_out, _ = module.preprocess_spark(df, model=self.scaler, train=False)

            # ===== PROMETHEUS METRICS =====
            record_count = df_out.count()
            PREPROCESS_RECORDS.labels(dataset=dataset).inc(record_count)
            PREPROCESS_BATCHES.labels(dataset=dataset).inc(1)
            
            # Per-class distribution (if 'attack' column exists)
            target_col = self.params.get('preprocessing', {}).get('target', 'attack')
            if target_col in df_out.columns:
                class_counts = df_out.groupBy(target_col).count().collect()
                for row in class_counts:
                    cls = str(row[target_col])
                    cnt = row['count']
                    PREPROCESS_RECORDS_BY_CLASS.labels(dataset=dataset, **{'class': cls}).inc(cnt)
                    PREPROCESS_RECORDS_LAST_BATCH.labels(dataset=dataset, **{'class': cls}).set(cnt)
            
            elapsed = time.time() - start_time
            PREPROCESS_BATCH_LATENCY.labels(dataset=dataset).observe(elapsed)

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

    def _save_scaler_artifact(self):
        """Spark guarda modelos directamente en S3A."""
        try:
            s3a_path = self._to_s3a_path(self.artifacts_dir)
            model_path = os.path.join(s3a_path, 'pipeline_model')
            
            with open("/tmp/pipeline_model.json", "w") as f:
                json.dump(self.scaler, f)

            self.s3.upload_file(
                "/tmp/pipeline_model.json",
                Bucket="k8s-mlops-platform-bucket",
                Key="v1/artifacts/pipeline_model.json"
            )
            self.logger.info(f'PipelineModel saved to {model_path}')
        except Exception as e:
            self.logger.error('Failed saving model to S3', exc_info=True)
            raise

    def load_scaler_artifact(self):
        """Carga el pipeline primero desde local; si no existe, lo baja de S3."""
        try:
            local_path = "/tmp/pipeline_model.json"

            # 1️⃣ Si existe en local, úsalo
            if os.path.exists(local_path):
                self.logger.info("Loading pipeline artifact from local cache")
                with open(local_path, "r") as f:
                    self.scaler = json.load(f)
                return self.scaler

            # 2️⃣ Si no existe, descargar desde S3
            self.logger.info("Local pipeline not found. Downloading from S3")

            if self.s3 is None:
                self._check_minio_connection()

            self.s3.download_file(
                "k8s-mlops-platform-bucket",
                "v1/artifacts/pipeline_model.json",
                local_path
            )

            with open(local_path, "r") as f:
                self.scaler = json.load(f)

            self.logger.info("Pipeline artifact downloaded from S3 and loaded")
            return self.scaler

        except Exception as e:
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
    from src.schemas.spark.schemas import schema_full as schema
    
    data_dir = "s3a://k8s-mlops-platform-bucket/v1/raw/" #Para Spark
    output_dir = "s3a://k8s-mlops-platform-bucket/v1/processed/" #Para Spark
    artifacts_dir = "s3a://k8s-mlops-platform-bucket/v1/artifacts/"

    preprocessing = SparkPreprocessing(
        schema=schema,
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

if __name__ == "__main__":
    main()
