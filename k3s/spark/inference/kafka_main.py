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

# ===== FUNCIONES STANDALONE PARA PREDICCIONES =====

def _predict_partition_ray_online(
    partition: Iterator[Row],
    config: Dict[str, Any],
) -> Iterator[Row]:
    """
    Online mode: sends each schema-converted row as {"raw": {...}} to Ray Serve.

    Spark ran kafka_to_schema_features (schema conversion only).
    Ray Serve runs the NumpyPipelineExecutor (DSL) + model prediction.
    One HTTP request per event.
    """
    url = config['url']
    timeout = config['timeout']
    id_column = config['id_column']
    pred_column = config['prediction_column']
    pred_key = config.get('prediction_key', 'predictions')

    with requests.Session() as session:
        for row in partition:
            event_id = row[id_column]
            raw_dict = row.asDict()
            try:
                response = session.post(url, json={"raw": raw_dict}, timeout=timeout)
                response.raise_for_status()
                preds = response.json().get(pred_key, [0])
                pred = preds[0] if preds else 0
                yield Row(**{id_column: event_id, pred_column: int(pred)})
            except Exception as e:
                print(f"[online] Error sending event {event_id!r} to Ray Serve: {e}")
                yield Row(**{id_column: event_id, pred_column: 0})



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

        # Cargar configuraciones desde params
        self._load_kafka_config()
        self._load_schema_config()
        self._load_converter_config()
        self._load_prediction_config()
        self._load_output_config()
        self._load_checkpoint_config()

        self.converter_function = None

        # Validaciones opcionales
        operational_config = self.params.get('operational', {})
        if operational_config.get('check_kafka_connection', True):
            self._check_kafka_connection()

        # Crear Spark session
        self.spark = self._create_spark_session()

        # Cargar módulos y funciones
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
        """Carga configuración de Kafka desde params."""        
        # Usar env vars con fallback a params.yaml
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
        self.kafka_input_topic = os.getenv('KAFKA_TOPIC')
        self.kafka_output_topic = os.getenv('KAFKA_TOPIC_OUTPUT')        
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')

        self.kafka_sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM','SCRAM-SHA-512')
        self.kafka_security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_PLAINTEXT')
        
        self.kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.scram.ScramLoginModule required '
            f'username="{self.kafka_username}" password="{self.kafka_password}";'
        )
        
        # Streaming options
        streaming_config = self.params.get('streaming', {})
        self.online_processing_time = streaming_config.get('online_processing_time', '100 milliseconds')
        self.starting_offsets = streaming_config.get('starting_offsets', 'latest')
        self.fail_on_data_loss = streaming_config.get('fail_on_data_loss', False)
        
        self.logger.info(f"✅ Kafka config: {self.kafka_input_topic} → {self.kafka_output_topic}")
    
    def _load_schema_config(self):
        """Carga configuración de schemas desde params."""
        schema_config = self.params.get('schemas', {})
        
        # Nombres de schemas
        self.input_schema_name = schema_config.get('input_schema', 'kafka_schema_features')
        self.features_schema_name = schema_config.get('features_schema', 'schema_features')
        self.output_schema_name = schema_config.get('output_schema', 'prediction_schema')
        
        # Cargar schemas reales usando el registry
        self.input_schema = get_schema(self.input_schema_name)
        self.output_schema = get_schema(self.output_schema_name)
        
        self.logger.info(
            f"✅ Schemas loaded: input={self.input_schema_name}, "
            f"output={self.output_schema_name}"
        )
    
    def _load_converter_config(self):
        """Carga configuración de conversores desde params."""
        converter_config = self.params.get('converters', {})
        self.converter_name = converter_config.get('kafka_to_features', 'kafka_to_schema_features')
        self.logger.info(f"✅ Converter configured: {self.converter_name}")
    
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
        """Carga configuración de predicción desde params."""
        pred_config = self.params.get('prediction', {})
        
        self.prediction_type = pred_config.get('type', 'ray_serve')
        
        # Configuración Ray Serve
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
        
        # Configuración de columnas
        columns_config = pred_config.get('columns', {})
        self.id_column = columns_config.get('id_column', 'event_id')
        self.prediction_column = columns_config.get('prediction_column', 'label')
        
        self.logger.info(f"✅ Prediction: {self.prediction_type} at {self.ray_url}")
    
    def _load_output_config(self):
        """Carga configuración de salida desde params."""
        output_config = self.params.get('output', {})
        
        self.output_format = output_config.get('format', 'json')
        self.output_key_column = output_config.get('key_column', 'event_id')
        self.output_value_columns = output_config.get('value_columns', ['timestamp', 'event_id', 'properties', 'label'])
        
        self.logger.info(f"✅ Output: format={self.output_format}")
    
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
        """Carga dinámicamente la función conversora."""
        try:
            self.converter_function = get_converter(self.converter_name)
            self.logger.info(f"✅ Converter function loaded: {self.converter_name}")
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
        """Crea Spark session con configuración desde params."""
        try:
            self.logger.info("Creating SparkSession with Kafka and S3 support")            
            builder = SparkSession.builder.appName(self.params['app_name'])

            # Configuración S3
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
        """Convierte schema usando el conversor configurado."""
        try:
            df_converted = self.converter_function(df)
            return df_converted
        except Exception as e:
            self.logger.error(f"Schema conversion failed: {e}")
            raise
    
    def predict_batch_online(self, df_converted: DataFrame) -> DataFrame:
        """
        Online mode: sends schema-converted rows to Ray Serve as raw events.

        Each row becomes {"raw": <schema_dict>}. Ray Serve runs the NumPy DSL
        executor and the model, returning one prediction per event.
        """
        pred_config = {
            'url': str(self.ray_url),
            'timeout': int(self.ray_timeout),
            'id_column': str(self.id_column),
            'prediction_column': str(self.prediction_column),
            'prediction_key': str(self.ray_pred_key),
        }
        predictions_rdd = df_converted.rdd.mapPartitions(
            lambda p: _predict_partition_ray_online(p, pred_config)
        )
        return self.spark.createDataFrame(predictions_rdd, schema=self.output_schema)

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

            # 1. Schema conversion (kafka_to_schema_features)
            df_features = self.convert_schema(batch_df)

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
                for cls in range(self.params.get('num_classes', 2)):
                    cnt = counts_by_cls.get(cls, 0)
                    PREDICTIONS_BY_CLASS_LAST_BATCH.labels(str(cls)).set(cnt)
                    if cnt:
                        PREDICTIONS_BY_CLASS_TOTAL.labels(str(cls)).inc(cnt)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not compute prediction class distribution: {e}")

            # 4. Join con datos originales
            output_df = batch_df.join(predictions_df, on=self.id_column, how='inner')

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
        """Escribe resultados a Kafka según configuración."""
        try:
            # Seleccionar columnas según configuración
            if isinstance(self.output_value_columns, list):
                columns_expr = ', '.join(self.output_value_columns)
            elif self.output_value_columns == 'all':
                columns_expr = '*'
            else:
                # Por defecto, usar las columnas hardcodeadas originales
                columns_expr = 'timestamp, event_id, properties, label'
            
            # Formatear salida
            if self.output_format == 'json':
                df_output = df.selectExpr(
                    f"CAST({self.output_key_column} AS STRING) AS key",
                    f"to_json(struct({columns_expr})) AS value"
                )
            else:
                df_output = df
            
            # Escribir
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
                .option("checkpointLocation", self.checkpoint_location)
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