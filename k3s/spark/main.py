"""
spark_preprocess_iceberg.py

Pipeline de preprocesamiento con trazabilidad completa y optimizaciones de cálculo:
- Lee raw data desde tabla Iceberg
- Aplica transformaciones DSL
- Genera tablas procesadas versionadas
- Mantiene metadata tracking con snapshots

OPTIMIZACIONES APLICADAS:
1. Eliminación de recomputación crítica (lectura única de raw data)
2. Persistencia inteligente con MEMORY_AND_DISK en DataFrames reutilizados
3. Eliminación de validaciones con count() que disparan el DAG
4. Configuraciones de cálculo optimizadas (AQE, shuffle partitions, broadcast)
5. Reparticionado estratégico antes de escritura a Iceberg
"""
from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
import boto3
import os
import time
import hashlib
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel
from prometheus_client import start_http_server
from src.prometheus import (
    PREPROCESS_RECORDS,
    PREPROCESS_BATCHES,
    PREPROCESS_RECORDS_BY_CLASS,
    PREPROCESS_RECORDS_LAST_BATCH,
    PREPROCESS_BATCH_LATENCY,
    PREPROCESS_BATCH_LATENCY_LAST,
    PREPROCESS_ERRORS,
    PREPROCESS_CURRENT_DATASET,
)
from src.dsl import Pipeline, PipelineModel
import tempfile
from src.schemas.spark.schema_registry import get_schema
from pyspark.sql.types import TimestampType, StringType, IntegerType, LongType, FloatType, DoubleType, BooleanType, DateType

class SparkPreprocessIceberg(BaseUtils):
    """
    Preprocesamiento con full trazabilidad y optimizaciones de cálculo:
    - Raw data desde Iceberg con lectura única
    - Processed tables versionadas
    - Metadata tracking con snapshots
    - Persistencia inteligente para evitar recomputación
    """
    def __init__(self, params_path: str):
        logger = create_logger('SparkPreprocessIceberg', 'spark_preprocess_iceberg.log')
        super().__init__(logger, params_path)
        
        self.params = self.load_params()
        self.spark_params = self.params.get('spark', {})
        self.preprocess_params = self.spark_params.get('preprocessing', {})
        
        # Iceberg configuration
        self.iceberg_cfg = self.params.get('iceberg_tables', {})
        # Configuración de tablas (prioridad: Env Var > params.yaml > hardcoded)
        self.raw_table = self.iceberg_cfg.get('raw', {}).get('full_name', 'iceberg.raw.network_traffic_raw')
        self.metadata_table = self.iceberg_cfg.get('metadata', {}).get('full_name', 'iceberg.metadata.preprocessing_artifacts')
        
        # Path al DSL
        self.dsl_path = self.preprocess_params.get('dsl_path', '/app/repo/k3s/spark/preprocess/dsl_001.yaml')
        
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
        # Prometheus metrics
        self._start_prometheus_server()

    def _load_artifacts_config(self):
        """Carga configuración de artifacts desde params."""
        artifacts_config = self.preprocess_params.get('artifacts', {})
        
        self.artifacts_bucket = artifacts_config.get('bucket', 'k8s-mlops-platform-bucket')
        self.pipelines_prefix = 'pipelines'
        self.artifacts_list = artifacts_config.get('archives', ['config.json', 'stages.json'])
        
        self.logger.info(f"Artifacts config: bucket={self.artifacts_bucket}, prefix={self.pipelines_prefix}")

    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server on port 8001."""
        port = int(os.getenv('PROMETHEUS_PORT', 8001))
        try:
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")

            # Pre-create expected labelsets so Grafana panels show train/val/test even if a split has 0 rows.
            for split in ("train", "val", "test", "full"):
                PREPROCESS_RECORDS.labels(dataset=split).inc(0)
                PREPROCESS_BATCHES.labels(dataset=split).inc(0)
                PREPROCESS_BATCH_LATENCY_LAST.labels(dataset=split).set(0)

                # Pre-create class labelsets (default num_classes)
                num_classes = self.spark_params.get('num_classes', 2)
                for c in range(max(num_classes, 1)):
                    PREPROCESS_RECORDS_LAST_BATCH.labels(dataset=split, **{'class': str(c)}).set(0)
        except Exception as e:
            self.logger.warning(f"Could not start Prometheus server: {e}")

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
        """
        Crea SparkSession con soporte para Iceberg y optimizaciones de cálculo.
        
        - spark.sql.shuffle.partitions = adaptive (usa AQE para determinar óptimo)
        - spark.sql.autoBroadcastJoinThreshold = 50MB (aumentado para evitar shuffles)
        - spark.sql.adaptive.advisoryPartitionSizeInBytes = 128MB (optimiza coalescencia)
        - spark.sql.adaptive.coalescePartitions.minPartitionSize = 16MB
        """
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
            
            # ============ OPTIMIZACIONES DE CÁLCULO ============
            # AQE ya está habilitado en YAML, pero reforzamos aquí
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            
            # Shuffle partitions: usar menos particiones para datasets pequeños
            # AQE ajustará dinámicamente, pero partimos de un valor más conservador
            .config("spark.sql.shuffle.partitions", "200")  # Default, AQE optimizará
            
            # Broadcast join threshold: aumentar a 50MB para evitar shuffles innecesarios
            # Datasets pequeños se benefician enormemente de broadcast joins
            .config("spark.sql.autoBroadcastJoinThreshold", str(50 * 1024 * 1024))
            
            # AQE: tamaño advisory de partición (128MB es un buen balance)
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", str(128 * 1024 * 1024))
            
            # AQE: tamaño mínimo de partición al coalescer (16MB evita particiones muy pequeñas)
            .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", str(16 * 1024 * 1024))
            
            # Skew join: threshold para detectar particiones sesgadas
            .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
            .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", str(256 * 1024 * 1024))
            
            .getOrCreate()
        )
        
        self.logger.info("SparkSession created with Iceberg catalog at %s", warehouse)
        self.logger.info("Computation optimizations enabled: AQE, broadcast threshold=50MB, adaptive shuffle")
        return spark

    def get_raw_snapshot_id(self) -> int:
        """
        Obtiene el snapshot ID actual de la tabla raw.
        Este snapshot representa el estado exacto de los datos crudos.
        """
        try:
            snapshots = self.spark.sql(
                f"SELECT snapshot_id FROM {self.raw_table}.snapshots ORDER BY committed_at DESC LIMIT 1"
            )
            snapshot_id = snapshots.collect()[0]['snapshot_id']
            self.logger.info(f"Raw table snapshot: {snapshot_id}")
            return snapshot_id
        except Exception as e:
            self.logger.error(f"Failed to get snapshot from {self.raw_table}: {e}", exc_info=True)
            raise

    def load_raw_data_by_date_range(self, start_date: str, end_date: str):
        """
        Lee datos de la tabla raw filtrando por rango de fechas.
        
        OPTIMIZACIÓN: Pushdown de predicados a Iceberg para leer solo datos necesarios.
        
        Args:
            start_date: fecha inicio (formato ISO: 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM:SS')
            end_date: fecha fin (formato ISO)
        
        Returns:
            DataFrame filtrado (NO persistido, lazy evaluation)
        """
        try:
            self.logger.info(f"Loading raw data: {start_date} to {end_date} using explicit schema")
            
            # Leer la tabla raw y forzar el esquema
            df = self.spark.table(self.raw_table).to(self.schema)
            
            # Robust timestamp conversion
            if 'timestamp' in df.columns:
                field = next((f for f in df.schema.fields if f.name == 'timestamp'), None)
                if field is not None and isinstance(field.dataType, TimestampType):
                    pass  # Ya es timestamp
                else:
                    df = df.withColumn(
                        "timestamp",
                        F.when(F.col("timestamp").cast("long").isNotNull(),
                               F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
                        .otherwise(F.to_timestamp(F.col("timestamp")))
                    )

            # Filtrar por rango de fechas (pushdown predicate a Iceberg)
            start_ts = F.to_timestamp(F.lit(start_date))
            end_ts = F.to_timestamp(F.lit(end_date))
            
            df_filtered = df.filter(
                (F.col("timestamp") >= start_ts) & (F.col("timestamp") <= end_ts)
            )

            # OPTIMIZACIÓN: NO hacer count() aquí, dejar lazy evaluation
            # La validación se hará cuando se materialice el DataFrame
            self.logger.info(f"DataFrame created (lazy) for range {start_date} to {end_date}")
            
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
            pipeline_hash = hash_obj.hexdigest()[:16]
            
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
            tmpdir = tempfile.mkdtemp(prefix="pipeline_model_")
            self.pipeline.save(tmpdir)
            
            s3_prefix = f"{self.pipelines_prefix}/{pipeline_hash}"
            
            for artifact_file in self.artifacts_list:
                local_path = os.path.join(tmpdir, artifact_file)
                s3_key = f"{s3_prefix}/{artifact_file}"
                
                self.s3.upload_file(local_path, Bucket=self.artifacts_bucket, Key=s3_key)
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
            
            for artifact_file in self.artifacts_list:
                s3_key = f"{s3_prefix}/{artifact_file}"
                local_path = os.path.join(tmpdir, artifact_file)
                self.s3.download_file(self.artifacts_bucket, s3_key, local_path)
            
            self.pipeline = PipelineModel.load(tmpdir)
            self.logger.info(f"Pipeline loaded from S3: {pipeline_hash}")
            
        except Exception as e:
            self.logger.error(f"Failed loading pipeline from S3: {e}", exc_info=True)
            raise

    def fit_pipeline(self, df, dsl_path: str) -> str:
        """
        Entrena el pipeline con los datos de train y retorna su hash.
        
        OPTIMIZACIÓN: Asume que df ya está persistido si es necesario (caller responsibility).
        NO hace validaciones con count() para evitar disparar el DAG innecesariamente.
        
        Args:
            df: DataFrame de train (puede estar persistido)
            dsl_path: path al DSL YAML
            
        Returns:
            pipeline_hash: hash del pipeline entrenado
        """
        try:
            self.logger.info(f"Fitting pipeline from DSL: {dsl_path}")
            
            # Cargar y entrenar pipeline
            base_pipeline = Pipeline.from_yaml(dsl_path)
            self.pipeline = base_pipeline.fit(df)
            
            # Calcular hash
            pipeline_hash = self.compute_pipeline_hash(dsl_path)
            
            # Guardar en S3
            self.save_pipeline_to_s3(pipeline_hash)
            
            self.logger.info(f"Pipeline fitted successfully with hash: {pipeline_hash}")
            return pipeline_hash
            
        except Exception as e:
            self.logger.error(f"Failed fitting pipeline: {e}", exc_info=True)
            raise

    def transform_data(self, df):
        """
        Aplica el pipeline entrenado a un DataFrame.
        
        OPTIMIZACIÓN: NO hace validaciones con count() para evitar disparar el DAG.
        
        Args:
            df: DataFrame a transformar (puede estar persistido)
            
        Returns:
            DataFrame transformado y con features seleccionadas (lazy)
        """
        try:
            if self.pipeline is None:
                raise ValueError("Pipeline not fitted. Call fit_pipeline first.")

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
        
        OPTIMIZACIÓN: Repartición estratégica antes de escritura para optimizar tamaño de archivos.
        
        Args:
            df: DataFrame procesado
            table_name: nombre de la tabla (sin namespace)
            namespace: namespace Iceberg (default: "processed")
        """
        try:
            full_table = f"iceberg.{namespace}.{table_name}"
            
            # Crear namespace si no existe
            self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg.{namespace}")
            
            self.logger.info(f"Writing processed table: {full_table}")

            processed_cfg = self.preprocess_params.get('processed', {}) if isinstance(self.preprocess_params, dict) else {}
            partition_spec = processed_cfg.get('partition')

            # OPTIMIZACIÓN: Repartición estratégica antes de escritura
            # Objetivo: archivos de ~128MB para balance entre paralelismo y overhead
            # Solo reparticionar si tenemos muchas particiones pequeñas
            num_partitions = df.rdd.getNumPartitions()
            self.logger.info(f"DataFrame has {num_partitions} partitions before write")
            
            # Heurística: si hay más de 400 particiones, coalescer a 200
            # AQE ya habrá optimizado, pero aseguramos no tener demasiadas particiones
            if num_partitions > 400:
                self.logger.info(f"Coalescing from {num_partitions} to 200 partitions")
                df = df.coalesce(200)
            
            # Si partition_spec está definido y tabla no existe, crear con particionado
            if partition_spec and not self.spark.catalog.tableExists(full_table):
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
            parts = self.metadata_table.split('.')
            if len(parts) == 3:
                namespace = parts[1]
            else:
                namespace = "metadata"
            
            self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg.{namespace}")
            
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
            train_cfg = self.splits_config.get('train', {})
            val_cfg = self.splits_config.get('val', {})
            test_cfg = self.splits_config.get('test', {})
            
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

            timestamp_cols = ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']
            for col in timestamp_cols:
                metadata_row = metadata_row.withColumn(col, F.to_timestamp(F.col(col)))
            
            metadata_row.writeTo(self.metadata_table).append()
            
            self.logger.info(f"Metadata record inserted: artifact_set_id={artifact_set_id}")
            
        except Exception as e:
            self.logger.error(f"Failed inserting metadata: {e}", exc_info=True)
            raise

    def run_preprocessing_pipeline(self, artifact_set_id: str):
        """
        Ejecuta el pipeline completo de preprocesamiento con optimizaciones de cálculo.
        
        OPTIMIZACIONES CRÍTICAS APLICADAS:
        1. LECTURA ÚNICA de raw data (antes se leía dos veces: train y full)
        2. PERSIST de df_full con MEMORY_AND_DISK (se reutiliza para train subset y transform)
        3. Filtrado en memoria para crear train subset (sin releer desde Iceberg)
        4. UNPERSIST explícito después de uso
        5. Sin validaciones count() innecesarias
        
        Workflow optimizado:
        1. Obtiene snapshot de raw data
        2. Lee FULL dataset UNA SOLA VEZ (train+val+test range)
        3. Persiste df_full (se usará dos veces: filtro para train + transform completo)
        4. Crea df_train como subset en memoria (filtro sobre df_full persistido)
        5. Entrena pipeline en df_train
        6. Transforma df_full (usa datos ya persistidos)
        7. Unpersiste df_full
        8. Escribe tabla procesada
        9. Actualiza metadata
        
        Args:
            artifact_set_id: ID único para este set de artifacts
        """
        try:
            start_time = time.time()
            PREPROCESS_CURRENT_DATASET.set(1)
            
            # 1. Obtener snapshot de raw data
            raw_snapshot = self.get_raw_snapshot_id()
            
            # 2. Extraer nombre del DSL
            dsl_name = os.path.basename(self.dsl_path)
            
            # 3. Asegurar que metadata table existe
            self.ensure_metadata_table_exists()
            
            # 4. Calcular rango completo (train + val + test)
            train_cfg = self.splits_config.get('train', {})
            val_cfg = self.splits_config.get('val', {})
            test_cfg = self.splits_config.get('test', {})
            
            date_starts = [d for d in [train_cfg.get('start'), val_cfg.get('start'), test_cfg.get('start')] if d]
            date_ends = [d for d in [train_cfg.get('end'), val_cfg.get('end'), test_cfg.get('end')] if d]
            
            if not date_starts or not date_ends:
                raise ValueError("Splits configuration must include start/end dates for train/val/test.")
            
            full_start = min(date_starts)
            full_end = max(date_ends)
            
            train_start = train_cfg.get('start')
            train_end = train_cfg.get('end')
            
            # ========== OPTIMIZACIÓN CRÍTICA: LECTURA ÚNICA ==========
            self.logger.info("=" * 60)
            self.logger.info("OPTIMIZATION: Loading FULL dataset ONCE (train+val+test)")
            self.logger.info(f"Full range: {full_start} to {full_end}")
            self.logger.info("=" * 60)
            
            # Leer full dataset UNA SOLA VEZ
            df_full = self.load_raw_data_by_date_range(full_start, full_end)
            
            # ========== OPTIMIZACIÓN: PERSIST ESTRATÉGICO ==========
            # Persistir con MEMORY_AND_DISK porque:
            # 1. Se usará para crear subset de train (filtro en memoria)
            # 2. Se usará para transform completo
            # 3. Si no cabe en memoria, spill a disco local del pod (emptyDir)
            # 4. Evita releer desde Iceberg/S3 dos veces
            self.logger.info("OPTIMIZATION: Persisting df_full with MEMORY_AND_DISK")
            df_full.persist(StorageLevel.MEMORY_AND_DISK)
            
            # Forzar materialización con una acción ligera (count es barato aquí porque solo se hace UNA VEZ)
            full_count = df_full.count()
            self.logger.info(f"Full dataset loaded and persisted: {full_count} rows")
            
            # ========== OPTIMIZACIÓN: TRAIN SUBSET EN MEMORIA ==========
            self.logger.info("=" * 60)
            self.logger.info("OPTIMIZATION: Creating train subset via in-memory filter")
            self.logger.info(f"Train range: {train_start} to {train_end}")
            self.logger.info("=" * 60)
            
            # Crear train como subset filtrando df_full (que ya está en memoria/disco)
            # Esto NO dispara lectura desde Iceberg, usa datos persistidos
            train_start_ts = F.to_timestamp(F.lit(train_start))
            train_end_ts = F.to_timestamp(F.lit(train_end))
            
            df_train = df_full.filter(
                (F.col("timestamp") >= train_start_ts) &
                (F.col("timestamp") <= train_end_ts)
            )
            
            # Opcional: persistir df_train si el pipeline.fit hace múltiples pasadas
            # Para la mayoría de pipelines, una pasada es suficiente, así que NO persistir
            # Si tu pipeline hace múltiples iteraciones sobre train, descomentar:
            # df_train.persist(StorageLevel.MEMORY_AND_DISK)
            
            train_count = df_train.count()
            self.logger.info(f"Train subset created: {train_count} rows (filtered from persisted full data)")

            # ===== PROMETHEUS METRICS (train subset) =====
            train_stage_end = time.time()
            train_elapsed = train_stage_end - start_time
            PREPROCESS_RECORDS.labels(dataset="train").inc(train_count)
            PREPROCESS_BATCHES.labels(dataset="train").inc(1)
            PREPROCESS_BATCH_LATENCY.labels(dataset="train").observe(train_elapsed)
            PREPROCESS_BATCH_LATENCY_LAST.labels(dataset="train").set(train_elapsed)

            # Per-class distribution (if target column exists)
            target_col = self.spark_params.get('target', 'attack')
            num_classes = self.spark_params.get('num_classes', 2)
            for c in range(max(num_classes, 1)):
                PREPROCESS_RECORDS_LAST_BATCH.labels(dataset="train", **{'class': str(c)}).set(0)
            if target_col in df_train.columns:
                class_counts = df_train.groupBy(target_col).count().collect()
                for row in class_counts:
                    cls = str(row[target_col])
                    cnt = row['count']
                    PREPROCESS_RECORDS_BY_CLASS.labels(dataset="train", **{'class': cls}).inc(cnt)
                    PREPROCESS_RECORDS_LAST_BATCH.labels(dataset="train", **{'class': cls}).set(cnt)
            
            # ========== FIT PIPELINE EN TRAIN ==========
            self.logger.info("=" * 60)
            self.logger.info("Fitting pipeline on train subset")
            self.logger.info("=" * 60)
            pipeline_hash = self.fit_pipeline(df_train, self.dsl_path)
            
            # Si df_train fue persistido, liberar ahora
            # df_train.unpersist()
            
            # ========== TRANSFORM FULL DATASET ==========
            self.logger.info("=" * 60)
            self.logger.info("OPTIMIZATION: Transforming FULL dataset (using persisted data)")
            self.logger.info("=" * 60)
            
            # Transform usa df_full que ya está persistido
            # NO dispara lectura desde Iceberg
            df_full_processed = self.transform_data(df_full)
            
            # ========== UNPERSIST df_full ==========
            # Liberar memoria/disco antes de write
            self.logger.info("OPTIMIZATION: Unpersisting df_full (no longer needed)")
            df_full.unpersist()
            
            # ========== WRITE PROCESSED TABLE ==========
            self.logger.info("=" * 60)
            self.logger.info("Writing processed table to Iceberg")
            self.logger.info("=" * 60)
            
            processed_table_name = f"{artifact_set_id}"
            self.write_processed_table(df_full_processed, processed_table_name)

            # ===== PROMETHEUS METRICS (full dataset) =====
            full_stage_end = time.time()
            full_elapsed = full_stage_end - train_stage_end
            PREPROCESS_RECORDS.labels(dataset="full").inc(full_count)
            PREPROCESS_BATCHES.labels(dataset="full").inc(1)
            PREPROCESS_BATCH_LATENCY.labels(dataset="full").observe(full_elapsed)
            PREPROCESS_BATCH_LATENCY_LAST.labels(dataset="full").set(full_elapsed)
            
            # ========== INSERT METADATA ==========
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
            self.logger.info(f"✓ Preprocessing pipeline completed in {elapsed:.2f}s")
            self.logger.info(f"✓ Artifact Set ID: {artifact_set_id}")
            self.logger.info(f"✓ Raw Snapshot: {raw_snapshot}")
            self.logger.info(f"✓ Pipeline Hash: {pipeline_hash}")
            self.logger.info(f"✓ Processed Table: {processed_table_ref}")
            self.logger.info(f"✓ Full dataset rows: {full_count}")
            self.logger.info(f"✓ Train subset rows: {train_count}")
            self.logger.info("=" * 60)
            PREPROCESS_CURRENT_DATASET.set(0)
            
        except Exception as e:
            PREPROCESS_ERRORS.labels(dataset="full", error_type=type(e).__name__).inc()
            PREPROCESS_CURRENT_DATASET.set(0)
            self.logger.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
            raise

def main():
    """
    Entry point del job de preprocesamiento.
    """
    params_path = os.getenv("PARAMS_PATH", "/app/repo/k3s/params.yaml")
    
    # Inicializar job (carga params internamente)
    job = SparkPreprocessIceberg(params_path=params_path)
    
    artifact_set_id = os.getenv("ARTIFACT_SET_ID")
    if not artifact_set_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dsl_basename = os.path.basename(job.dsl_path).replace('.yaml', '')
        artifact_set_id = f"{dsl_basename}_{timestamp}"
    
    job.run_preprocessing_pipeline(
        artifact_set_id=artifact_set_id
    )

if __name__ == "__main__":
    main()