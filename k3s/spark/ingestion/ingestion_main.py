# spark_ingest_raw.py
from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import os
import time
from pyspark.sql import SparkSession
from src.schemas.spark.schema_registry import get_schema
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, LongType, DoubleType, StringType

class SparkIngestionRaw(BaseUtils):
    """
    Ingest job: lee parquet crudo desde S3 y lo escribe a una tabla Iceberg
    en el catálogo 'iceberg.raw'.

    Ordena solo los datos recién leídos por 'timestamp' antes de escribir.
    Append-only; no modifica datos ya existentes.
    """

    def __init__(self, params_path: str, raw_s3_path: str, table_name: str):
        logger = create_logger('SparkIngestRaw', 'spark_ingest_raw.log')
        super().__init__(logger, params_path)
        self.params = self.load_params().get('spark', {})
        self.raw_s3_path = self._to_s3a_path(raw_s3_path)
        self.table_name = table_name
        self.schema = get_schema(self.params.get('schemas', {}).get('full_schema'))
        self.s3 = None
        self.spark = self._create_spark_session()

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
            self.logger.info('S3 connection verified. Buckets: %s', bucket_names)
        except Exception as e:
            self.logger.error('S3 connection failed: %s', str(e), exc_info=True)
            raise

    @staticmethod
    def _to_s3a_path(path: str) -> str:
        if path.startswith('s3a://'):
            return path
        if path.startswith('s3://'):
            return 's3a://' + path[len('s3://'):]
        if path.startswith('https://') or path.startswith('http://'):
            try:
                parts = path.split('amazonaws.com/')
                if len(parts) == 2:
                    host = parts[0]
                    prefix = parts[1]
                    bucket = host.split('//')[1].split('.')[0]
                    return f"s3a://{bucket}/{prefix}"
            except Exception:
                pass
        return path

    def _create_spark_session(self):
        self._check_minio_connection()
        warehouse = self.params.get('iceberg', {}).get(
            'warehouse',
            's3a://k8s-mlops-platform-bucket/warehouse'
        )
        spark = (
            SparkSession.builder
            .appName(self.params.get('app_name', 'spark-ingest-raw'))
            .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.path.style.access", "false")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            .config("spark.hadoop.fs.s3a.experimental.fadvise", "random")
            .config("spark.sql.files.maxPartitionBytes",
                    int(self.params.get('read_batch_size', 256)) * 1024 * 1024)
            .config("spark.sql.extensions",
                    "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.iceberg",
                    "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.iceberg.type", "hadoop")
            .config("spark.sql.catalog.iceberg.warehouse", warehouse)
            .getOrCreate()
        )
        self.logger.info("SparkSession created (Iceberg catalog at %s)", warehouse)
        return spark

    def load_data(self, path: str):
        # Read without forcing schema to avoid parquet timestamp mismatch
        df = self.spark.read.parquet(path)
        self.logger.info("Loaded raw DF | partitions=%s", df.rdd.getNumPartitions())
        return df

    def _coerce_to_schema(self, df):
        """Coerce incoming DataFrame to expected schema types with safe casts."""
        for field in self.schema.fields:
            name = field.name
            if name not in df.columns:
                continue

            dtype = field.dataType
            if isinstance(dtype, TimestampType):
                ts_col = F.col(name).cast("double")
                seconds = (
                    F.when(ts_col > F.lit(1e14), ts_col / F.lit(1e6))
                     .when(ts_col > F.lit(1e11), ts_col / F.lit(1e3))
                     .otherwise(ts_col)
                )
                df = df.withColumn(name, F.to_timestamp(F.from_unixtime(seconds)))
            elif isinstance(dtype, LongType):
                df = df.withColumn(name, F.col(name).cast("long"))
            elif isinstance(dtype, DoubleType):
                df = df.withColumn(name, F.col(name).cast("double"))
            elif isinstance(dtype, StringType):
                df = df.withColumn(name, F.col(name).cast("string"))
        return df

    def table_exists(self, full_table_name: str) -> bool:
        try:
            return self.spark.catalog.tableExists(full_table_name)
        except Exception:
            return False

    def build_final_df(self, df):
        """
        Ordena solo los datos recién leídos por 'timestamp'.
        Convierte epoch a timestamp cuando sea necesario para orden correcto.
        """
        # Coerce to expected schema and ensure timestamp is valid
        df_with_ts = self._coerce_to_schema(df)
        df_ordered = df_with_ts.orderBy(F.col("timestamp").asc_nulls_last())
        return df_ordered

    def write_raw_to_iceberg(self, df, namespace: str = "raw"):
        full_table = f"iceberg.{namespace}.{self.table_name}"
        self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg.{namespace}")

        force_recreate = os.getenv("FORCE_RECREATE_RAW", "false").lower() == "true"
        if force_recreate and self.table_exists(full_table):
            self.logger.warning("FORCE_RECREATE_RAW=true -> dropping table %s", full_table)
            self.spark.sql(f"DROP TABLE IF EXISTS {full_table}")

        if not self.table_exists(full_table):
            self.logger.info("Table does not exist — creating %s", full_table)
            df.writeTo(full_table).using("iceberg").option("format-version", "2").create()
        else:
            self.logger.info("Table exists — appending to %s", full_table)
            df.writeTo(full_table).using("iceberg").option("format-version", "2").append()

def main():
    raw_https_path = "https://k8s-mlops-platform-bucket.s3.us-east-2.amazonaws.com/raw/"
    raw_path = SparkIngestionRaw._to_s3a_path(raw_https_path)

    table_name = os.getenv("ICEBERG_RAW_TABLE", "network_traffic_raw")
    params_path = os.getenv("PARAMS_PATH", "/app/repo/k3s/params.yaml")

    job = SparkIngestionRaw(params_path=params_path, raw_s3_path=raw_path, table_name=table_name)

    start = time.time()
    try:
        df_raw = job.load_data(job.raw_s3_path)
        df_final = job.build_final_df(df_raw)  # Ordena solo lo recién leído
        job.write_raw_to_iceberg(df_final, namespace="raw")
        job.logger.info("Ingest finished successfully")
    except Exception:
        job.logger.error("Ingest job failed", exc_info=True)
        raise
    finally:
        job.logger.info("Ingest job elapsed: %.2fs", time.time() - start)


if __name__ == "__main__":
    main()
