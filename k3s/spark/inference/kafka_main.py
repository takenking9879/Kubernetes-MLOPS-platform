"""
Kafka-Spark Inference Pipeline — router mode.

Spark actúa como router de alta velocidad entre Kafka y Ray Serve:
  Kafka → Spark (schema conversion only) → Ray Serve (DSL NumPy + inferencia) → Kafka

Ray Serve carga el NumpyPipelineExecutor + modelo vía MLflow alias.
Spark no descarga artifacts, no ejecuta DSL, no hace preprocessing.
"""
import os
import time
from typing import Dict, Optional, Iterator, List, Any
from urllib.parse import urlparse
from confluent_kafka import Consumer, KafkaException
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.functions import from_json, col, to_json, struct
import requests
from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
from src.schemas.spark.schema_registry import get_schema
from src.converters.spark_kafka_helper import get_converter

# ===== PROMETHEUS METRICS =====
from prometheus_client import start_http_server
from src.prometheus import (
    LATENCY_INFERENCE,
    LATENCY_TOTAL_BATCH,
    LATENCY_KAFKA_WRITE,
    BATCH_RECORDS_TOTAL,
    BATCH_ERRORS_TOTAL,
    INFERENCE_LATENCY_SUMMARY,
    PREDICTIONS_BY_CLASS_TOTAL,
    PREDICTIONS_BY_CLASS_LAST_BATCH,
)

# ─── System constants (not user-configurable) ─────────────────────────────────

# Schema names are internal system identifiers registered in SchemaRegistry.
# They never vary per dataset run.
_KAFKA_INPUT_SCHEMA  = "kafka_schema_features"
_FEATURES_SCHEMA     = "schema_features"
_KAFKA_OUTPUT_SCHEMA = "prediction_schema"

# Output column naming conventions — system defaults, not user decisions.
_PREDICTION_COLUMN  = "label"                # name of the predicted class column
_INTERNAL_ID_COLUMN = "__internal_row_id"    # Spark-Ray join key + Kafka key fallback
_DEFAULT_NUM_CLASSES = 6        # for Prometheus label initialisation

_APP_NAME = "kafka-spark-inference"


# ===== FUNCIONES STANDALONE PARA PREDICCIONES =====

def _predict_partition_ray_online(
    partition: Iterator[Row],
    config: Dict[str, Any],
) -> Iterator[Row]:
    """
    Online mode: sends each schema-converted row as {"raw": {...}} to Ray Serve.

    Spark ran kafka_to_schema_features (schema conversion only).
    Ray Serve runs the NumpyPipelineExecutor (DSL) + model prediction.
    Sends batched events per partition to reduce HTTP overhead and queueing.
    """
    url = config['url']
    timeout = config['timeout']
    internal_id = config.get('internal_id_column', '__internal_row_id')
    pred_column = config['prediction_column']
    pred_key = config.get('prediction_key', 'predictions')
    batch_size = max(1, int(config.get('batch_size', 100)))
    max_retries = max(0, int(config.get('max_retries', 3)))

    def _chunk_rows(rows: Iterator[Row], size: int) -> Iterator[List[Row]]:
        chunk: List[Row] = []
        for item in rows:
            chunk.append(item)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _emit_default(rows_batch: List[Row]) -> Iterator[Row]:
        for row in rows_batch:
            yield Row(**{internal_id: row[internal_id], pred_column: 0})

    with requests.Session() as session:
        for rows_batch in _chunk_rows(partition, batch_size):
            raw_batch = [row.asDict() for row in rows_batch]
            row_ids = [row[internal_id] for row in rows_batch]

            delivered = False
            for attempt in range(max_retries + 1):
                try:
                    response = session.post(url, json={"raw_batch": raw_batch}, timeout=timeout)
                    response.raise_for_status()
                    body = response.json()
                    preds = body.get(pred_key, [])

                    if not isinstance(preds, list) or len(preds) != len(row_ids):
                        raise ValueError(
                            f"Invalid prediction response size. expected={len(row_ids)}, got={len(preds) if isinstance(preds, list) else 'non-list'}"
                        )

                    for row_id, pred in zip(row_ids, preds):
                        yield Row(**{internal_id: row_id, pred_column: int(pred)})

                    delivered = True
                    break
                except Exception as e:
                    if attempt >= max_retries:
                        print(
                            f"[online] Batched request failed after {max_retries + 1} attempts "
                            f"(events={len(rows_batch)}): {e}"
                        )
            if not delivered:
                yield from _emit_default(rows_batch)



# ===== PIPELINE GENÉRICO =====

class KafkaSparkInference(BaseUtils):
    """
    Kafka-Spark router: schema conversion only, all ML work in Ray Serve.

    Pipeline:
      Kafka → convert_schema() → predict_batch_online() [HTTP → Ray Serve] → Kafka
    """

    def __init__(self, params_path: str):
        logger = create_logger('kafka_spark_inference', 'kafka_spark_inference.log')
        super().__init__(logger, params_path)
        self.params_full = self.load_params()
        self.params = self.params_full.get('spark', {})
        # ===== START PROMETHEUS METRICS SERVER =====
        self._start_prometheus_server()

        # Cargar configuraciones
        self._load_kafka_config()
        self._load_schemas()
        self._load_prediction_config()
        self._load_checkpoint_config()

        self.converter_function = None

        # Validaciones opcionales
        operational_config = self.params.get('operational', {})
        if operational_config.get('check_kafka_connection', True):
            self._check_kafka_connection()

        # Crear Spark session
        self.spark = self._create_spark_session()

        # Cargar converter (hardcoded — only one converter exists)
        self._load_converter_function()

        self.logger.info(
            "Router mode: Spark converts schema and forwards events to Ray Serve. "
            "DSL preprocessing + prediction run in Ray Serve (NumpyPipelineExecutor)."
        )

    # ===== PROMETHEUS SERVER =====

    def _start_prometheus_server(self):
        """Start Prometheus HTTP server on port 8000 for custom metrics."""
        try:
            prometheus_port = int(os.getenv('PROMETHEUS_PORT', '8000'))
            start_http_server(prometheus_port)
            self.logger.info(f"✅ Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not start Prometheus server: {e}")

    # ===== CONFIGURATION LOADERS =====

    def _load_kafka_config(self):
        """Carga configuración de Kafka desde env vars y params."""
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
        self.kafka_input_topic = os.getenv('KAFKA_TOPIC')
        self.kafka_output_topic = os.getenv('KAFKA_TOPIC_OUTPUT')
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')

        self.kafka_sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', 'SCRAM-SHA-512')
        self.kafka_security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_PLAINTEXT')

        self.kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.scram.ScramLoginModule required '
            f'username="{self.kafka_username}" password="{self.kafka_password}";'
        )

        # Streaming options (legitimately variable — kept in static params.yaml)
        streaming_config = self.params.get('streaming', {})
        self.online_processing_time = streaming_config.get('online_processing_time', '100 milliseconds')
        self.starting_offsets = streaming_config.get('starting_offsets', 'latest')
        self.fail_on_data_loss = streaming_config.get('fail_on_data_loss', False)

        self.logger.info(f"✅ Kafka config: {self.kafka_input_topic} → {self.kafka_output_topic}")

    def _load_schemas(self):
        """Load Kafka input/output schemas from the schema registry.

        Schema names are system constants — not user-configurable.
        """
        self.input_schema  = get_schema(_KAFKA_INPUT_SCHEMA)
        self.output_schema = get_schema(_KAFKA_OUTPUT_SCHEMA)
        self.logger.info(
            "✅ Schemas loaded: input=%s, output=%s",
            _KAFKA_INPUT_SCHEMA, _KAFKA_OUTPUT_SCHEMA,
        )

    def _read_id_field_from_s3(self) -> Optional[str]:
        """Download raw.yaml from RAW_SCHEMA_S3_PATH and read the optional id_field annotation.

        Returns None if the env var is unset, the download fails, or id_field is absent.
        None signals that __internal_row_id should be used as the Kafka key fallback.
        """
        s3_path = os.getenv('RAW_SCHEMA_S3_PATH', '')
        if not s3_path:
            return None
        try:
            import boto3
            import yaml
            import re
            match = re.match(r"s3://([^/]+)/(.+)", s3_path)
            if not match:
                return None
            bucket, key = match.group(1), match.group(2)
            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-2"),
            )
            content = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            schema = yaml.safe_load(content) or {}
            id_field = schema.get("id_field")  # None when annotation is absent
            if id_field:
                self.logger.info("✅ id_field='%s' read from %s", id_field, s3_path)
            return str(id_field) if id_field else None
        except Exception as exc:
            self.logger.warning("⚠️ Could not read id_field from RAW_SCHEMA_S3_PATH: %s", exc)
            return None

    def _get_raw_top_level_fields(self) -> List[str]:
        """Return the top-level field names of the Kafka input schema.

        Used to build the Kafka output value_columns without hard-coding them.
        Falls back to a safe default list if introspection fails.
        """
        schema = self.input_schema
        try:
            # pyspark StructType exposes .fieldNames() or .names
            if hasattr(schema, 'fieldNames'):
                return list(schema.fieldNames())
            if hasattr(schema, 'names'):
                return list(schema.names)
            if hasattr(schema, 'fields'):
                return [f.name for f in schema.fields]
        except Exception:
            pass
        # Minimal safe fallback — no id assumption
        return ['timestamp', 'properties']

    def _normalize_serve_url(self, raw_url: str) -> str:
        """Normaliza URL de Ray Serve para tolerar http implícito y /infer/ trailing slash."""
        candidate = (raw_url or '').strip()
        if not candidate:
            return 'http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer'

        if '://' not in candidate:
            candidate = f"http://{candidate}"

        parsed = urlparse(candidate)
        scheme = parsed.scheme or 'http'
        netloc = parsed.netloc
        path = parsed.path or ''

        if not netloc and path:
            netloc = path
            path = ''

        path = path.rstrip('/')
        if path in ('', '/'):
            path = '/infer'

        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized = f"{normalized}?{parsed.query}"
        return normalized

    def _load_prediction_config(self):
        """Carga configuración de predicción desde params.

        - prediction_column: system constant (_PREDICTION_COLUMN = 'label').
        - id_column: dataset-specific; read from raw.yaml id_field annotation
          via RAW_SCHEMA_S3_PATH env var, or overridden by KAFKA_ID_COLUMN.
        """
        pred_config = self.params.get('prediction', {})

        self.prediction_type = pred_config.get('type', 'ray_serve')

        # Ray Serve endpoint config (legitimately variable)
        if self.prediction_type == 'ray_serve':
            ray_config = pred_config.get('ray_serve', {})
            raw_ray_url = os.getenv(
                'RAY_SERVE_URL',
                ray_config.get('url', 'http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer')
            )
            self.ray_url = self._normalize_serve_url(raw_ray_url)
            self.ray_batch_size = int(ray_config.get('batch_size', 100))
            self.ray_timeout = int(ray_config.get('timeout', 30))
            self.ray_max_retries = ray_config.get('max_retries', 3)
            self.ray_payload_format = ray_config.get('request_payload_format', 'list')
            self.ray_pred_key = ray_config.get('prediction_key', 'predictions')

        # prediction_column: always 'label' — system naming convention
        self.prediction_column = _PREDICTION_COLUMN

        # id_column: optional — P1: KAFKA_ID_COLUMN env, P2: raw.yaml id_field, P3: None → __internal_row_id
        self.id_column: Optional[str] = os.getenv('KAFKA_ID_COLUMN') or self._read_id_field_from_s3()

        # Derive output_value_columns: top-level raw schema fields + prediction column
        self._raw_top_level_fields = self._get_raw_top_level_fields()

        self.logger.info(
            "✅ Prediction: type=%s, url=%s, kafka_key=%s, prediction_column=%s",
            self.prediction_type, self.ray_url,
            self.id_column or f"<{_INTERNAL_ID_COLUMN}>",
            self.prediction_column,
        )

    def _load_checkpoint_config(self):
        """Carga configuración de checkpoint desde params."""
        checkpoint_config = self.params.get('checkpoint', {})
        self.checkpoint_location = os.getenv(
            'SPARK_CHECKPOINT_LOCATION',
            checkpoint_config.get('location', 's3a://k8s-mlops-platform-bucket/checkpoints/kafka-spark-inference')
        )
        self.logger.info(f"✅ Checkpoint: {self.checkpoint_location}")

    # ===== MODULE LOADERS =====

    def _load_converter_function(self):
        """Loads the schema converter.

        kafka_to_schema_features is the only converter in the registry.
        No dynamic lookup needed.
        """
        try:
            self.converter_function = get_converter("kafka_to_schema_features")
            self.logger.info("✅ Converter function loaded: kafka_to_schema_features")
        except Exception as e:
            self.logger.error(f"Failed to load converter: {e}")
            raise

    # ===== CONNECTION CHECKS =====

    def _check_kafka_connection(self):
        """Validación ligera de conectividad Kafka."""
        try:
            self.logger.info(f"Checking Kafka connection: {self.kafka_bootstrap_servers}")

            conf = {
                'bootstrap.servers': self.kafka_bootstrap_servers,
                'group.id': 'kafka-connection-test',
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': False,
                'security.protocol': self.kafka_security_protocol,
                'sasl.mechanism': self.kafka_sasl_mechanism,
                'sasl.username': self.kafka_username,
                'sasl.password': self.kafka_password,
                'socket.timeout.ms': 5000,
                'api.version.request.timeout.ms': 5000
            }

            consumer = Consumer(conf)
            metadata = consumer.list_topics(timeout=5)
            topics = metadata.topics
            consumer.close()

            if topics:
                self.logger.info(f"✅ Kafka connected: {len(topics)} topics available")
            else:
                self.logger.warning("⚠️ Kafka connected but no topics found")

        except KafkaException as e:
            self.logger.error(f"Kafka connection failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during Kafka check: {e}")
            raise

    # ===== SPARK SESSION =====

    def _create_spark_session(self):
        """Crea Spark session con configuración S3 y Kafka."""
        try:
            self.logger.info("Creating SparkSession with Kafka and S3 support")
            app_name = os.getenv('SPARK_APP_NAME', _APP_NAME)
            builder = SparkSession.builder.appName(app_name)

            builder = (
                builder
                .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
                .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
                .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
                .config("spark.hadoop.fs.s3a.path.style.access", "false")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.committer.name", "magic")
                .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
                .config("spark.sql.streaming.schemaInference", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.streaming.metricsEnabled", "false")
            )

            spark = builder.getOrCreate()
            self.logger.info("✅ SparkSession created")
            return spark

        except Exception as e:
            self.logger.error(f"Failed to create SparkSession: {e}")
            raise

    # ===== PIPELINE STEPS =====

    def read_from_kafka_online(self) -> DataFrame:
        """
        Lee Kafka stream para online=True (micro-batch de baja latencia).
        Sin maxOffsetsPerTrigger para drenar la cola tan rápido como sea posible.
        """
        try:
            self.logger.info("Reading from Kafka (online mode): %s", self.kafka_input_topic)
            df = (
                self.spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
                .option("subscribe", self.kafka_input_topic)
                .option("startingOffsets", self.starting_offsets)
                .option("kafka.security.protocol", self.kafka_security_protocol)
                .option("kafka.sasl.mechanism", self.kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", self.kafka_sasl_jaas_config)
                .option("failOnDataLoss", str(self.fail_on_data_loss).lower())
                .load()
            )
            parsed_df = (
                df.selectExpr("CAST(value AS STRING)")
                .select(from_json(col("value"), self.input_schema).alias("data"))
                .select("data.*")
            )
            self.logger.info("✅ Kafka online stream configured (no maxOffsetsPerTrigger)")
            return parsed_df
        except Exception as e:
            self.logger.error("Failed to read from Kafka (online): %s", e)
            raise

    def convert_schema(self, df: DataFrame) -> DataFrame:
        """Convierte schema usando el conversor configurado y añade ID interno para el join."""
        try:
            from pyspark.sql.functions import monotonically_increasing_id
            df_converted = self.converter_function(df)
            # Añadir ID interno para asegurar un join robusto con Ray Serve sin depender de id_field
            return df_converted.withColumn(_INTERNAL_ID_COLUMN, monotonically_increasing_id())
        except Exception as e:
            self.logger.error(f"Schema conversion failed: {e}")
            raise

    def predict_batch_online(self, df_converted: DataFrame) -> DataFrame:
        """
        Online mode: sends schema-converted rows to Ray Serve as raw events.

        Includes an internal row identifier to ensure robust joins even
        if the dataset doesn't have a unique ID.
        """
        from pyspark.sql.types import StructType, StructField, LongType, IntegerType
        pred_config = {
            'url': str(self.ray_url),
            'timeout': int(self.ray_timeout),
            'batch_size': int(self.ray_batch_size),
            'max_retries': int(self.ray_max_retries),
            'internal_id_column': _INTERNAL_ID_COLUMN,
            'prediction_column': str(self.prediction_column),
            'prediction_key': str(self.ray_pred_key),
        }
        predictions_rdd = df_converted.rdd.mapPartitions(
            lambda p: _predict_partition_ray_online(p, pred_config)
        )
        # Internal schema for the prediction join
        join_schema = StructType([
            StructField(_INTERNAL_ID_COLUMN, LongType(), False),
            StructField(self.prediction_column, IntegerType(), False)
        ])
        return self.spark.createDataFrame(predictions_rdd, schema=join_schema)

    def process_batch(self, batch_df: DataFrame, batch_id: int):
        """
        Procesa un micro-batch completo.
        Pipeline: convert_schema → predict_batch_online [Ray Serve] → join → write

        Includes Prometheus metrics for latency tracking.
        """
        batch_start_time = time.time()

        try:
            self.logger.info(f"Processing batch {batch_id}")

            record_count = batch_df.count()
            BATCH_RECORDS_TOTAL.set(record_count)

            if record_count == 0:
                self.logger.info(f"⏭️ Batch {batch_id} is empty, skipping")
                return

            # 1. Schema conversion (kafka_to_schema_features) + Internal ID generation
            df_features = self.convert_schema(batch_df)

            # Para poder hacer el join final, batch_df también necesita el ID interno
            from pyspark.sql.functions import monotonically_increasing_id
            batch_df_with_id = batch_df.withColumn(_INTERNAL_ID_COLUMN, monotonically_increasing_id())

            # 2. Predict via Ray Serve (DSL NumPy + model inference)
            inference_start = time.time()
            predictions_df = self.predict_batch_online(df_features)
            inference_latency = time.time() - inference_start
            LATENCY_INFERENCE.set(inference_latency)
            INFERENCE_LATENCY_SUMMARY.observe(inference_latency)

            # 3. Distribution of predicted classes (for attack monitoring)
            try:
                rows = (
                    predictions_df
                    .selectExpr(f"CAST({self.prediction_column} AS INT) AS cls")
                    .groupBy("cls")
                    .count()
                    .collect()
                )
                counts_by_cls = {int(r["cls"]): int(r["count"]) for r in rows if r["cls"] is not None}
                for cls in range(_DEFAULT_NUM_CLASSES):
                    cnt = counts_by_cls.get(cls, 0)
                    PREDICTIONS_BY_CLASS_LAST_BATCH.labels(str(cls)).set(cnt)
                    if cnt:
                        PREDICTIONS_BY_CLASS_TOTAL.labels(str(cls)).inc(cnt)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not compute prediction class distribution: {e}")

            # 4. Join con datos originales usando el ID interno temporal
            output_df = batch_df_with_id.join(predictions_df, on=_INTERNAL_ID_COLUMN, how='inner')

            # 5. Escribir a Kafka
            kafka_write_start = time.time()
            self._write_to_kafka(output_df)
            kafka_write_latency = time.time() - kafka_write_start
            LATENCY_KAFKA_WRITE.set(kafka_write_latency)

            total_latency = time.time() - batch_start_time
            LATENCY_TOTAL_BATCH.set(total_latency)

            self.logger.info(
                f"✅ Batch {batch_id} completed | Records: {record_count} | "
                f"Inference: {inference_latency:.3f}s | Kafka write: {kafka_write_latency:.3f}s | "
                f"Total: {total_latency:.3f}s"
            )

        except Exception as e:
            BATCH_ERRORS_TOTAL.inc()
            self.logger.error(f"❌ Batch {batch_id} failed: {e}", exc_info=True)
            raise

    def _write_to_kafka(self, df: DataFrame):
        """Escribe resultados a Kafka.

        Kafka key resolution (in priority order):
          P1 — KAFKA_ID_COLUMN env var   (explicit deploy-time override)
          P2 — id_field from raw.yaml    (user-declared semantic business key)
          P3 — __internal_row_id         (system fallback — always valid)
        value = JSON of all top-level raw schema fields + prediction column
        """
        try:
            kafka_key_col = self.id_column or _INTERNAL_ID_COLUMN
            value_cols = ', '.join(self._raw_top_level_fields + [self.prediction_column])
            df_output = df.selectExpr(
                f"CAST({kafka_key_col} AS STRING) AS key",
                f"to_json(struct({value_cols})) AS value"
            )

            (
                df_output.write
                .format("kafka")
                .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
                .option("topic", self.kafka_output_topic)
                .option("kafka.security.protocol", self.kafka_security_protocol)
                .option("kafka.sasl.mechanism", self.kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", self.kafka_sasl_jaas_config)
                .save()
            )

        except Exception as e:
            self.logger.error(f"Failed to write to Kafka: {e}")
            raise

    # ===== MAIN PIPELINE =====

    def run_inference(self):
        """
        Ejecuta el pipeline de inferencia como router de alta velocidad.
        Fast micro-batch: foreachBatch + short processingTime, sin maxOffsetsPerTrigger.
        """
        try:
            self.logger.info("🚀 Kafka-Spark Inference Pipeline — router mode")
            df_stream = self.read_from_kafka_online()
            query = (
                df_stream
                .writeStream
                .foreachBatch(self.process_batch)
                # .option("checkpointLocation", self.checkpoint_location)
                .outputMode("append")
                .trigger(processingTime=self.online_processing_time)
                .start()
            )
            self.logger.info("📍 Checkpoint: %s", self.checkpoint_location)
            self.logger.info("📥 %s → 📤 %s", self.kafka_input_topic, self.kafka_output_topic)
            self.logger.info("⚡ processingTime=%s", self.online_processing_time)
            self.logger.info("🧮 Spark router: schema conversion → Ray Serve (DSL + inference)")
            self.logger.info("⏳ Awaiting termination...")
            query.awaitTermination()

        except Exception as e:
            self.logger.error("❌ Pipeline failed: %s", e, exc_info=True)
            raise


# ===== ENTRY POINT =====
def main():
    """Punto de entrada de la aplicación."""
    params_path = '/app/repo/k3s/params.yaml'
    pipeline = KafkaSparkInference(params_path=params_path)
    pipeline.run_inference()

if __name__ == "__main__":
    main()
