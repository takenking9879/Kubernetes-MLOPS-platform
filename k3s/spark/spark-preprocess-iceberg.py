"""
spark_preprocess_iceberg.py

Pipeline de preprocesamiento con trazabilidad completa:
- Lee raw data desde tabla Iceberg
- Aplica transformaciones DSL
- Genera tablas procesadas versionadas
- Mantiene metadata tracking con snapshots
"""

from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import os
import time
import hashlib
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.dsl import Pipeline, PipelineModel
import tempfile
from src.schemas.spark.schema_registry import get_schema
from pyspark.sql.types import TimestampType

class SparkPreprocessIceberg(BaseUtils):
    """
    Preprocesamiento con full trazabilidad:
    - Raw data desde Iceberg
    - Processed tables versionadas
    - Metadata tracking con snapshots
    """

    def __init__(self, params_path: str, raw_table: str, metadata_table: str):
        logger = create_logger('SparkPreprocessIceberg', 'spark_preprocess_iceberg.log')
        super().__init__(logger, params_path)
        
        self.params = self.load_params()
        self.spark_params = self.params.get('spark', {})
        self.preprocess_params = self.spark_params.get('preprocessing', {})
        
        # Configuración de tablas
        self.raw_table = raw_table  # e.g., "iceberg.raw.network_traffic_raw"
        self.metadata_table = metadata_table  # e.g., "iceberg.metadata.preprocessing_artifacts"
        
        # Splits configuration
        self.splits_config = self.params.get('splits', {})
        
        # Schema y pipeline
        self.schema = get_schema(self.spark_params.get('schemas', {}).get('full_schema'))
        self.pipeline = None
        
        # S3 y Spark
        self.s3 = None
        self.spark = self._create_spark_session()
        
        # Pipeline artifacts config
        self._load_artifacts_config()

    def _load_artifacts_config(self):
        """Carga configuración de artifacts desde params."""
        artifacts_config = self.preprocess_params.get('artifacts', {})
        
        self.artifacts_bucket = artifacts_config.get('bucket', 'k8s-mlops-platform-bucket')
        self.pipelines_prefix = 'pipelines'  # s3://bucket/pipelines/pipeline_hash/
        self.artifacts_list = artifacts_config.get('archives', ['config.json', 'stages.json'])
        
        self.logger.info(f"Artifacts config: bucket={self.artifacts_bucket}, prefix={self.pipelines_prefix}")

    def _check_s3_connection(self):
        """Verifica conexión a S3/MinIO."""
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

    def _create_spark_session(self):
        """Crea SparkSession con soporte para Iceberg."""
        self._check_s3_connection()
        
        warehouse = self.spark_params.get('iceberg', {}).get(
            'warehouse',
            's3a://k8s-mlops-platform-bucket/warehouse'
        )
        
        spark = (
            SparkSession.builder
            .appName(self.spark_params.get('app_name', 'spark-preprocess-iceberg'))
            .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            .config("spark.hadoop.fs.s3a.experimental.fadvise", "random")
            .config("spark.sql.files.maxPartitionBytes",
                    int(self.spark_params.get('read_batch_size', 256)) * 1024 * 1024)
            # Iceberg extensions
            .config("spark.sql.extensions",
                    "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.iceberg",
                    "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.iceberg.type", "hadoop")
            .config("spark.sql.catalog.iceberg.warehouse", warehouse)
            # Magic committer for writes
            .config("spark.hadoop.fs.s3a.committer.name", "magic")
            .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
            .getOrCreate()
        )
        
        self.logger.info("SparkSession created with Iceberg catalog at %s", warehouse)
        return spark

    def get_raw_snapshot_id(self) -> int:
        """
        Obtiene el snapshot ID actual de la tabla raw.
        Este snapshot representa el estado exacto de los datos crudos.
        """
        try:
            snapshots = self.spark.sql(f"SELECT snapshot_id FROM {self.raw_table}.snapshots ORDER BY committed_at DESC LIMIT 1")
            # Debug: sample recent snapshots
            try:
                snaps_sample = [r.asDict() for r in snapshots.limit(5).collect()]
                self.logger.debug("Raw snapshots sample: %s", snaps_sample)
            except Exception:
                self.logger.debug("Could not collect snapshots sample", exc_info=True)

            snapshot_id = snapshots.collect()[0]['snapshot_id']
            self.logger.info(f"Raw table snapshot: {snapshot_id}")
            return snapshot_id
        except Exception as e:
            self.logger.error(f"Failed to get snapshot from {self.raw_table}: {e}", exc_info=True)
            raise

    def load_raw_data_by_date_range(self, start_date: str, end_date: str):
        """
        Lee datos de la tabla raw filtrando por rango de fechas.
        
        Args:
            start_date: fecha inicio (formato ISO: 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM:SS')
            end_date: fecha fin (formato ISO)
        
        Returns:
            DataFrame filtrado
        """
        try:
            self.logger.info(f"Loading raw data: {start_date} to {end_date}")
            
            # Leer toda la tabla raw
            df = self.spark.table(self.raw_table)
            # Debug: schema/columns and small sample of raw table
            try:
                self.logger.debug("Raw table schema: %s", df.schema.simpleString())
            except Exception:
                self.logger.debug("Failed to read raw schema", exc_info=True)
            try:
                self.logger.debug("Raw table columns: %s", df.columns)
            except Exception:
                self.logger.debug("Failed to read raw columns", exc_info=True)
            try:
                raw_sample = df.limit(5).collect()
                self.logger.debug("Raw table sample (up to 5 rows): %s", [r.asDict() for r in raw_sample])
            except Exception:
                self.logger.debug("Failed to collect raw sample", exc_info=True)
            
            # Robust timestamp conversion:
            # - if numeric epoch -> convert from_unixtime
            # - else try to_timestamp on string values
            if 'timestamp' in df.columns:
                field = next((f for f in df.schema.fields if f.name == 'timestamp'), None)
                if field is not None and isinstance(field.dataType, TimestampType):
                    # already timestamp type
                    pass
                else:
                    # Log original timestamp dtype/sample before conversion
                    try:
                        self.logger.debug("Timestamp field type before conversion: %s", [f.dataType.simpleString() for f in df.schema.fields if f.name == 'timestamp'])
                    except Exception:
                        self.logger.debug("Failed to inspect timestamp field type", exc_info=True)

                    df = df.withColumn(
                        "timestamp",
                        F.when(F.col("timestamp").cast("long").isNotNull(),
                               F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
                        .otherwise(F.to_timestamp(F.col("timestamp")))
                    )

                    # Debug: inspect timestamp column after conversion (sample)
                    try:
                        ts_sample = df.select("timestamp").limit(5).collect()
                        self.logger.debug("Timestamp column sample after conversion: %s", [r.asDict() for r in ts_sample])
                    except Exception:
                        self.logger.debug("Failed to collect timestamp sample after conversion", exc_info=True)

            # Filtrar por rango de fechas usando to_timestamp on literals for safe comparison
            start_ts = F.to_timestamp(F.lit(start_date))
            end_ts = F.to_timestamp(F.lit(end_date))

            df_filtered = df.filter((F.col("timestamp") >= start_ts) & (F.col("timestamp") <= end_ts))

            # Avoid full count (expensive). Use a cheap existence check for logging.
            has_any = df_filtered.limit(1).count()
            self.logger.info(f"Loaded sample presence={has_any} for range {start_date} to {end_date}")
            self.logger.debug("Filter bounds: start=%s end=%s", start_date, end_date)
            try:
                self.logger.debug("Filtered df schema: %s", df_filtered.schema.simpleString())
            except Exception:
                self.logger.debug("Failed to read filtered schema", exc_info=True)
            try:
                filtered_sample = df_filtered.limit(5).collect()
                self.logger.debug("Filtered sample (up to 5 rows): %s", [r.asDict() for r in filtered_sample])
            except Exception:
                self.logger.debug("Failed to collect filtered sample", exc_info=True)
            
            return df_filtered
            
        except Exception as e:
            self.logger.error(f"Failed loading raw data: {e}", exc_info=True)
            raise

    def compute_pipeline_hash(self, dsl_path: str) -> str:
        """
        Calcula hash del pipeline DSL para versionamiento.
        
        Args:
            dsl_path: path al archivo DSL YAML
            
        Returns:
            hash hexadecimal del contenido del DSL
        """
        try:
            with open(dsl_path, 'r') as f:
                dsl_content = f.read()
            
            hash_obj = hashlib.sha256(dsl_content.encode('utf-8'))
            pipeline_hash = hash_obj.hexdigest()[:16]  # Primeros 16 caracteres
            
            self.logger.info(f"Pipeline hash computed: {pipeline_hash} for {dsl_path}")
            return pipeline_hash
            
        except Exception as e:
            self.logger.error(f"Failed computing pipeline hash: {e}", exc_info=True)
            raise

    def save_pipeline_to_s3(self, pipeline_hash: str):
        """
        Guarda el pipeline entrenado en S3 bajo pipelines/{pipeline_hash}/
        
        Args:
            pipeline_hash: hash único del pipeline
        """
        try:
            # Crear directorio temporal
            tmpdir = tempfile.mkdtemp(prefix="pipeline_model_")
            
            # Guardar pipeline localmente
            self.pipeline.save(tmpdir)
            
            # Subir cada archivo a S3
            s3_prefix = f"{self.pipelines_prefix}/{pipeline_hash}"
            
            for artifact_file in self.artifacts_list:
                local_path = os.path.join(tmpdir, artifact_file)
                s3_key = f"{s3_prefix}/{artifact_file}"
                
                self.s3.upload_file(
                    local_path,
                    Bucket=self.artifacts_bucket,
                    Key=s3_key
                )
                self.logger.info(f"Uploaded {artifact_file} to s3://{self.artifacts_bucket}/{s3_key}")
            
            self.logger.info(f"Pipeline saved to S3: s3://{self.artifacts_bucket}/{s3_prefix}/")
            
        except Exception as e:
            self.logger.error(f"Failed saving pipeline to S3: {e}", exc_info=True)
            raise

    def load_pipeline_from_s3(self, pipeline_hash: str):
        """
        Carga un pipeline desde S3 usando su hash.
        
        Args:
            pipeline_hash: hash del pipeline a cargar
        """
        try:
            tmpdir = tempfile.mkdtemp(prefix="pipeline_load_")
            s3_prefix = f"{self.pipelines_prefix}/{pipeline_hash}"
            
            # Descargar artifacts
            for artifact_file in self.artifacts_list:
                s3_key = f"{s3_prefix}/{artifact_file}"
                local_path = os.path.join(tmpdir, artifact_file)
                
                self.s3.download_file(
                    self.artifacts_bucket,
                    s3_key,
                    local_path
                )
            
            # Cargar pipeline
            self.pipeline = PipelineModel.load(tmpdir)
            self.logger.info(f"Pipeline loaded from S3: {pipeline_hash}")
            
        except Exception as e:
            self.logger.error(f"Failed loading pipeline from S3: {e}", exc_info=True)
            raise

    def fit_pipeline(self, df, dsl_path: str) -> str:
        """
        Entrena el pipeline con los datos de train y retorna su hash.
        
        Args:
            df: DataFrame de train
            dsl_path: path al DSL YAML
            
        Returns:
            pipeline_hash: hash del pipeline entrenado
        """
        try:
            self.logger.info(f"Fitting pipeline from DSL: {dsl_path}")

            # Quick debug: ensure train DF has data and log schema/sample
            try:
                train_has_any = df.limit(1).count()
            except Exception as e:
                self.logger.error(f"Failed to peek into train DataFrame: {e}", exc_info=True)
                self.logger.debug('Train peek failed: could not inspect train DF before fit', exc_info=True)
                raise

            if train_has_any == 0:
                msg = "Train DataFrame is empty. Aborting fit."
                self.logger.error(msg)
                try:
                    self.logger.debug("Train DF schema when empty: %s", df.schema.simpleString())
                except Exception:
                    self.logger.debug("Failed to read train schema when empty", exc_info=True)
                try:
                    sample = df.limit(5).collect()
                    self.logger.debug("Train DF sample when empty (up to 5 rows): %s", [r.asDict() for r in sample])
                except Exception:
                    self.logger.debug("Failed to collect train sample when empty", exc_info=True)
                raise ValueError(msg)

            try:
                self.logger.debug("Train DF schema before fit: %s", df.schema.simpleString())
            except Exception:
                self.logger.debug("Failed to read train schema before fit", exc_info=True)
            try:
                train_sample = df.limit(5).collect()
                self.logger.debug("Train DF sample (up to 5 rows) before fit: %s", [r.asDict() for r in train_sample])
            except Exception:
                self.logger.debug("Failed to collect train sample before fit", exc_info=True)

            # Cargar y entrenar pipeline
            base_pipeline = Pipeline.from_yaml(dsl_path)
            self.pipeline = base_pipeline.fit(df)
            
            # Calcular hash
            pipeline_hash = self.compute_pipeline_hash(dsl_path)
            
            # Guardar en S3
            self.save_pipeline_to_s3(pipeline_hash)
            
            return pipeline_hash
            
        except Exception as e:
            self.logger.error(f"Failed fitting pipeline: {e}", exc_info=True)
            raise

    def transform_data(self, df):
        """
        Aplica el pipeline entrenado a un DataFrame.
        
        Args:
            df: DataFrame a transformar
            
        Returns:
            DataFrame transformado y con features seleccionadas
        """
        try:
            if self.pipeline is None:
                raise ValueError("Pipeline not fitted. Call fit_pipeline first.")

            # Check input df and log schema/sample
            try:
                has_any = df.limit(1).count()
            except Exception as e:
                self.logger.error(f"Failed to inspect DataFrame before transform: {e}", exc_info=True)
                self.logger.debug('Transform peek failed: could not inspect DF before transform', exc_info=True)
                raise

            if has_any == 0:
                msg = "Input DataFrame to transform is empty. Aborting transform."
                self.logger.error(msg)
                try:
                    self.logger.debug("Transform input schema when empty: %s", df.schema.simpleString())
                except Exception:
                    self.logger.debug("Failed to read transform input schema when empty", exc_info=True)
                try:
                    sample = df.limit(5).collect()
                    self.logger.debug("Transform input sample when empty (up to 5 rows): %s", [r.asDict() for r in sample])
                except Exception:
                    self.logger.debug("Failed to collect transform input sample when empty", exc_info=True)
                raise ValueError(msg)

            try:
                self.logger.debug("Transform input schema: %s", df.schema.simpleString())
            except Exception:
                self.logger.debug("Failed to read transform input schema", exc_info=True)
            try:
                transform_sample = df.limit(5).collect()
                self.logger.debug("Transform input sample (up to 5 rows): %s", [r.asDict() for r in transform_sample])
            except Exception:
                self.logger.debug("Failed to collect transform input sample", exc_info=True)

            self.logger.info("Applying pipeline transformations...")
            df_processed = self.pipeline.transform(df)
            df_out = self.pipeline.select_features(df_processed)
            
            return df_out
            
        except Exception as e:
            self.logger.error(f"Failed transforming data: {e}", exc_info=True)
            raise

    def write_processed_table(self, df, table_name: str, namespace: str = "processed"):
        """
        Escribe DataFrame procesado a una tabla Iceberg.
        
        Args:
            df: DataFrame procesado
            table_name: nombre de la tabla (sin namespace)
            namespace: namespace Iceberg (default: "processed")
        """
        try:
            full_table = f"iceberg.{namespace}.{table_name}"
            
            # Crear namespace si no existe
            self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg.{namespace}")
            
            # Escribir tabla. Si se indica particionado en params y la tabla no existe,
            # crear tabla con particionado y luego hacer append.
            self.logger.info(f"Writing processed table: {full_table}")

            processed_cfg = self.preprocess_params.get('processed', {}) if isinstance(self.preprocess_params, dict) else {}
            partition_spec = processed_cfg.get('partition')

            # If partition specified and table does not exist, create table with partition spec
            if partition_spec and not self.spark.catalog.tableExists(full_table):
                # Build simple DDL from dataframe schema
                from pyspark.sql.types import StringType, IntegerType, LongType, FloatType, DoubleType, BooleanType, TimestampType, DateType

                def spark_type(field):
                    t = field.dataType
                    if isinstance(t, StringType):
                        return 'STRING'
                    if isinstance(t, IntegerType):
                        return 'INT'
                    if isinstance(t, LongType):
                        return 'BIGINT'
                    if isinstance(t, FloatType):
                        return 'FLOAT'
                    if isinstance(t, DoubleType):
                        return 'DOUBLE'
                    if isinstance(t, BooleanType):
                        return 'BOOLEAN'
                    if isinstance(t, TimestampType):
                        return 'TIMESTAMP'
                    if isinstance(t, DateType):
                        return 'DATE'
                    return 'STRING'

                cols = ",\n".join([f"{fld.name} {spark_type(fld)}" for fld in df.schema.fields])

                create_sql = f"CREATE TABLE {full_table} (\n{cols}\n) USING iceberg PARTITIONED BY ({partition_spec}) OPTIONS ('format-version'='2')"
                self.logger.info(f"Creating partitioned table with DDL: {create_sql}")
                self.spark.sql(create_sql)

                # Insert data
                df.writeTo(full_table).using("iceberg").option("format-version", "2").append()
            else:
                # Default behavior: create or replace
                df.writeTo(full_table) \
                  .using("iceberg") \
                  .option("format-version", "2") \
                  .createOrReplace()

            self.logger.info(f"Processed table written: {full_table}")
            
        except Exception as e:
            self.logger.error(f"Failed writing processed table: {e}", exc_info=True)
            raise

    def ensure_metadata_table_exists(self):
        """
        Crea la tabla de metadata si no existe.
        
        Schema:
        - artifact_set_id: UUID único
        - created_at: timestamp de creación
        - raw_snapshot: snapshot ID de raw data
        - processed_table_name: nombre de tabla procesada
        - dsl_name: nombre del DSL usado
        - pipeline_hash: hash del pipeline en S3
        - train_start, train_end: rango train
        - val_start, val_end: rango val
        - test_start, test_end: rango test
        """
        try:
            # Parsear metadata_table
            parts = self.metadata_table.split('.')
            if len(parts) == 3:
                namespace = parts[1]
            else:
                namespace = "metadata"
            
            # Crear namespace
            self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg.{namespace}")
            
            # Verificar si tabla existe
            if not self.spark.catalog.tableExists(self.metadata_table):
                self.logger.info(f"Creating metadata table: {self.metadata_table}")
                
                create_sql = f"""
                CREATE TABLE {self.metadata_table} (
                    artifact_set_id STRING,
                    created_at TIMESTAMP,
                    raw_snapshot BIGINT,
                    processed_table_name STRING,
                    dsl_name STRING,
                    pipeline_hash STRING,
                    train_start TIMESTAMP,
                    train_end TIMESTAMP,
                    val_start TIMESTAMP,
                    val_end TIMESTAMP,
                    test_start TIMESTAMP,
                    test_end TIMESTAMP
                ) USING iceberg
                OPTIONS ('format-version'='2')
                """
                
                self.spark.sql(create_sql)
                self.logger.info(f"Metadata table created: {self.metadata_table}")
            else:
                self.logger.info(f"Metadata table already exists: {self.metadata_table}")
                
        except Exception as e:
            self.logger.error(f"Failed ensuring metadata table: {e}", exc_info=True)
            raise

    def insert_metadata_record(
        self,
        artifact_set_id: str,
        raw_snapshot: int,
        processed_table_name: str,
        dsl_name: str,
        pipeline_hash: str
    ):
        """
        Inserta un registro en la tabla de metadata.
        
        Args:
            artifact_set_id: UUID del artifact set
            raw_snapshot: snapshot ID de raw data
            processed_table_name: nombre completo de tabla procesada
            dsl_name: nombre del DSL
            pipeline_hash: hash del pipeline
        """
        try:
            # Obtener rangos de fechas de params
            train_cfg = self.splits_config.get('train', {})
            val_cfg = self.splits_config.get('val', {})
            test_cfg = self.splits_config.get('test', {})
            
            # Crear registro con casting explícito a TIMESTAMP para los rangos
            metadata_row = self.spark.createDataFrame([{
                'artifact_set_id': artifact_set_id,
                'created_at': datetime.now(),
                'raw_snapshot': raw_snapshot,
                'processed_table_name': processed_table_name,
                'dsl_name': dsl_name,
                'pipeline_hash': pipeline_hash,
                'train_start': train_cfg.get('start'),
                'train_end': train_cfg.get('end'),
                'val_start': val_cfg.get('start'),
                'val_end': val_cfg.get('end'),
                'test_start': test_cfg.get('start'),
                'test_end': test_cfg.get('end')
            }])

            # Aplicar casting explícito para asegurar compatibilidad con el esquema de la tabla Iceberg
            timestamp_cols = ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']
            for col in timestamp_cols:
                metadata_row = metadata_row.withColumn(col, F.to_timestamp(F.col(col)))
            
            # Insertar en metadata table
            metadata_row.writeTo(self.metadata_table).append()
            
            self.logger.info(f"Metadata record inserted: artifact_set_id={artifact_set_id}")
            
        except Exception as e:
            self.logger.error(f"Failed inserting metadata: {e}", exc_info=True)
            raise

    def run_preprocessing_pipeline(self, dsl_path: str, artifact_set_id: str):
        """
        Ejecuta el pipeline completo de preprocesamiento.
        
        Workflow:
        1. Obtiene snapshot de raw data
        2. Lee datos por split (train/val/test) según rangos
        3. Entrena pipeline en train
        4. Transforma TODOS los datos (train+val+test)
        5. Escribe UNA tabla procesada
        6. Actualiza metadata
        
        Args:
            dsl_path: path al archivo DSL
            artifact_set_id: ID único para este set de artifacts
        """
        try:
            start_time = time.time()
            
            # 1. Obtener snapshot de raw data
            raw_snapshot = self.get_raw_snapshot_id()
            
            # 2. Extraer nombre del DSL
            dsl_name = os.path.basename(dsl_path)
            
            # 3. Asegurar que metadata table existe
            self.ensure_metadata_table_exists()
            
            # 4. Procesar TRAIN (fit del pipeline)
            self.logger.info("=" * 60)
            self.logger.info("Processing TRAIN split")
            self.logger.info("=" * 60)
            
            train_cfg = self.splits_config.get('train', {})
            df_train = self.load_raw_data_by_date_range(
                train_cfg.get('start'),
                train_cfg.get('end')
            )
            
            # Fit pipeline en train
            pipeline_hash = self.fit_pipeline(df_train, dsl_path)

            # 5. Procesar TODOS los datos (train + val + test)
            self.logger.info("=" * 60)
            self.logger.info("Processing FULL dataset (train+val+test)")
            self.logger.info("=" * 60)

            val_cfg = self.splits_config.get('val', {})
            test_cfg = self.splits_config.get('test', {})

            date_starts = [d for d in [train_cfg.get('start'), val_cfg.get('start'), test_cfg.get('start')] if d]
            date_ends = [d for d in [train_cfg.get('end'), val_cfg.get('end'), test_cfg.get('end')] if d]

            if not date_starts or not date_ends:
                raise ValueError("Splits configuration must include start/end dates for train/val/test.")

            full_start = min(date_starts)
            full_end = max(date_ends)

            df_full = self.load_raw_data_by_date_range(full_start, full_end)
            df_full_processed = self.transform_data(df_full)

            # Write single processed table
            processed_table_name = f"{artifact_set_id}"
            self.write_processed_table(df_full_processed, processed_table_name)
            
            # 7. Insertar metadata
            # Usamos el nombre de la tabla única procesada como referencia principal
            processed_table_ref = f"iceberg.processed.{artifact_set_id}"
            
            self.insert_metadata_record(
                artifact_set_id=artifact_set_id,
                raw_snapshot=raw_snapshot,
                processed_table_name=processed_table_ref,
                dsl_name=dsl_name,
                pipeline_hash=pipeline_hash
            )
            
            elapsed = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info(f"Preprocessing pipeline completed in {elapsed:.2f}s")
            self.logger.info(f"Artifact Set ID: {artifact_set_id}")
            self.logger.info(f"Raw Snapshot: {raw_snapshot}")
            self.logger.info(f"Pipeline Hash: {pipeline_hash}")
            self.logger.info(f"Processed Table: {processed_table_ref}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
            raise


def main():
    """
    Entry point del job de preprocesamiento.
    
    Variables de entorno esperadas:
    - RAW_TABLE: nombre de tabla raw Iceberg (default: iceberg.raw.network_traffic_raw)
    - METADATA_TABLE: nombre de tabla metadata (default: iceberg.metadata.preprocessing_artifacts)
    - DSL_PATH: path al DSL YAML
    - ARTIFACT_SET_ID: ID único para este set (default: auto-generado con timestamp)
    - PARAMS_PATH: path a params.yaml
    """
    
    raw_table = os.getenv("RAW_TABLE", "iceberg.raw.network_traffic_raw")
    metadata_table = os.getenv("METADATA_TABLE", "iceberg.metadata.preprocessing_artifacts")
    dsl_path = os.getenv("DSL_PATH", "/app/repo/k3s/spark/preprocess/dsl_001.yaml")
    params_path = os.getenv("PARAMS_PATH", "/app/repo/k3s/params.yaml")
    
    # Generar artifact_set_id si no se provee
    artifact_set_id = os.getenv("ARTIFACT_SET_ID")
    if not artifact_set_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dsl_basename = os.path.basename(dsl_path).replace('.yaml', '')
        artifact_set_id = f"{dsl_basename}_{timestamp}"
    
    # Crear job
    job = SparkPreprocessIceberg(
        params_path=params_path,
        raw_table=raw_table,
        metadata_table=metadata_table
    )
    
    # Ejecutar pipeline
    job.run_preprocessing_pipeline(
        dsl_path=dsl_path,
        artifact_set_id=artifact_set_id
    )


if __name__ == "__main__":
    main()