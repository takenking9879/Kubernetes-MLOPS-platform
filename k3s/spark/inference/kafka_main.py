"""
Generic Kafka-Spark Inference Pipeline.
Completamente configurable vía params.yaml sin hardcodear lógica específica.

Refactorizado desde KafkaSparkInference para soportar múltiples datasets y modelos.
"""
import os
import time
from typing import Dict, Optional, Iterator, List, Any
from src.dsl import PipelineModel
from confluent_kafka import Consumer, KafkaException
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.functions import from_json, col, to_json, struct
import boto3
import requests
from src.utils.baseclass import BaseUtils
from src.utils.logger import create_logger
from src.schemas.spark.schema_registry import get_schema
from src.converters.spark_kafka_helper import get_converter

# ===== PROMETHEUS METRICS =====
from prometheus_client import start_http_server
from src.prometheus import (
    LATENCY_PREPROCESS,
    LATENCY_INFERENCE,
    LATENCY_TOTAL_BATCH,
    BATCH_RECORDS_TOTAL,
    BATCH_ERRORS_TOTAL,
    INFERENCE_LATENCY_SUMMARY,
    PREDICTIONS_BY_CLASS_TOTAL,
    PREDICTIONS_BY_CLASS_LAST_BATCH,
)

# ===== FUNCIÓN STANDALONE PARA PREDICCIONES =====

def _predict_partition_ray(
    partition: Iterator[Row], 
    config: Dict[str, Any]
) -> Iterator[Row]:
    """
    Función standalone para procesar particiones con predicciones Ray Serve.
    Extraída para evitar PicklingError.
    
    Args:
        partition: Iterador de Rows de Spark
        config: Diccionario con configuración (primitivos, no objetos)
    """
    url = config['url']
    batch_size = config['batch_size']
    timeout = config['timeout']
    id_column = config['id_column']
    pred_column = config['prediction_column']
    payload_format = config.get('payload_format', 'list')
    pred_key = config.get('prediction_key', 'predictions')
    
    batch_ids: List[Any] = []
    batch_payload: List[Any] = []
    
    def send_batch():
        """Helper para enviar batch y generar resultados."""
        if not batch_payload:
            return
        
        try:
            response = requests.post(url, json={"data": batch_payload}, timeout=timeout)
            response.raise_for_status()
            preds = response.json()[pred_key]
            
            if len(preds) != len(batch_ids):
                raise ValueError(f"Prediction count mismatch: {len(preds)} != {len(batch_ids)}")
            
            for i, pred in enumerate(preds):
                yield Row(**{id_column: batch_ids[i], pred_column: int(pred)})
        
        except requests.exceptions.RequestException as e:
            # Log error pero continuar procesamiento
            print(f"Error sending batch to Ray Serve: {e}")
            # Retornar predicciones por defecto (0)
            for batch_id in batch_ids:
                yield Row(**{id_column: batch_id, pred_column: 0})
    
    for row in partition:
        # Asume que event_id es la última columna (agregada por preprocessing)
        event_id = row[-1]
        payload_row = list(row)[:-1]
        
        batch_ids.append(event_id)
        
        if payload_format == 'list':
            batch_payload.append(payload_row)
        elif payload_format == 'dict':
            # Convertir a dict si es necesario (menos eficiente)
            batch_payload.append({f"f{i}": v for i, v in enumerate(payload_row)})
        else:
            batch_payload.append(payload_row)
        
        if len(batch_payload) >= batch_size:
            yield from send_batch()
            batch_ids = []
            batch_payload = []
    
    # Procesar batch restante
    if batch_ids:
        yield from send_batch()


# ===== PIPELINE GENÉRICO =====

class KafkaSparkInference(BaseUtils):
    """
    Pipeline genérico de inferencia Kafka-Spark.
    Todo configurable vía params.yaml.
    
    Cambios vs versión original:
    - Schemas cargados dinámicamente por nombre
    - Conversores cargados dinámicamente
    - Preprocessing module configurable
    - Output configurable
    - Artifacts configurable
    """
    
    def __init__(self, params_path: str):
        logger = create_logger('kafka_spark_inference', 'kafka_spark_inference.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['spark']
        # ===== START PROMETHEUS METRICS SERVER =====
        self._start_prometheus_server()
        
        # Cargar configuraciones desde params
        self._load_kafka_config()
        self._load_schema_config()
        self._load_converter_config()
        self._load_artifacts_config()
        self._load_prediction_config()
        self._load_output_config()
        self._load_checkpoint_config()
        
        # Estado interno
        self.s3 = None
        self.artifacts = None
        self.preprocessing_function = None
        self.converter_function = None
        
        # Validaciones opcionales
        operational_config = self.params.get('operational', {})
        if operational_config.get('check_kafka_connection', True):
            self._check_kafka_connection()
        
        if operational_config.get('check_minio_connection', True):
            self._check_minio_connection()
        
        # Cargar módulos y funciones
        self.pipeline = self.load_pipeline_artifact()
        self._load_converter_function()
        # Crear Spark session
        self.spark = self._create_spark_session()
    
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
        self.max_offsets_per_trigger = streaming_config.get('max_offsets_per_trigger', 10000)
        self.processing_time = streaming_config.get('processing_time', '10 seconds')
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
    
    def _load_artifacts_config(self):
        try:
            """Carga configuración de artifacts desde params."""
            artifacts_config = self.params.get('artifacts', {})
            
            # Configuración de artifacts
            self.artifacts_source = artifacts_config.get('source', 's3')
            self.artifacts_bucket = artifacts_config.get('bucket', 'k8s-mlops-platform-bucket')
            self.artifacts_key = artifacts_config.get('key', 'v1/artifacts')
            self.artifacts_list = artifacts_config.get('archives', ['config.json', 'stages.json'])
            self.logger.info(f"✅ Artifacts configured: archives={self.artifacts_list} from {self.artifacts_source}")
        except Exception as e:
            self.logger.error('Failed loading artifacts config: %s', str(e), exc_info=True)
            raise
    
    def _load_prediction_config(self):
        """Carga configuración de predicción desde params."""
        pred_config = self.params.get('prediction', {})
        
        self.prediction_type = pred_config.get('type', 'ray_serve')
        
        # Configuración Ray Serve
        if self.prediction_type == 'ray_serve':
            ray_config = pred_config.get('ray_serve', {})
            self.ray_url = os.getenv(
                'RAY_SERVE_URL',
                ray_config.get('url', 'http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer')
            )
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
    
    def _check_minio_connection(self):
        """Validación de conectividad S3/MinIO."""
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-2'),
            )
            buckets = self.s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            self.logger.info(f"✅ S3 connected: {len(bucket_names)} buckets")
        except Exception as e:
            self.logger.error(f"S3 connection failed: {e}")
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
            )
            
            spark = builder.getOrCreate()
            self.logger.info("✅ SparkSession created")
            return spark
            
        except Exception as e:
            self.logger.error(f"Failed to create SparkSession: {e}")
            raise
    
    # ===== ARTIFACT LOADING =====
    def load_pipeline_artifact(self):
        """Carga el PipelineModel desde local; si no existe, lo descarga de S3."""
        try:

            tmpdir = os.path.join("/tmp", "pipeline_model")
            if self.s3 is None:
                self._check_minio_connection()

            os.makedirs(tmpdir, exist_ok=True)

            for i in self.artifacts_list:
                self.s3.download_file(
                    self.artifacts_bucket,
                    os.path.join(self.artifacts_key, i),
                    os.path.join(tmpdir, i)
                )

            pipeline = PipelineModel.load(tmpdir)
            self.logger.info("Pipeline artifact downloaded from S3 and loaded")
            return pipeline
        except Exception:
            self.logger.error("Failed loading pipeline artifact", exc_info=True)
            raise
    
    # ===== PIPELINE STEPS =====
    
    def read_from_kafka(self) -> DataFrame:
        """Lee stream de Kafka con schema configurado."""
        try:
            self.logger.info(f"Reading from Kafka: {self.kafka_input_topic}")
            
            df = (
                self.spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
                .option("subscribe", self.kafka_input_topic)
                .option("startingOffsets", self.starting_offsets)
                .option("kafka.security.protocol", self.kafka_security_protocol)
                .option("kafka.sasl.mechanism", self.kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", self.kafka_sasl_jaas_config)
                .option("maxOffsetsPerTrigger", str(self.max_offsets_per_trigger))
                .option("failOnDataLoss", str(self.fail_on_data_loss).lower())
                .load()
            )
            
            # Parsear con schema configurado
            parsed_df = (
                df.selectExpr("CAST(value AS STRING)")
                .select(from_json(col("value"), self.input_schema).alias("data"))
                .select("data.*")
            )
            
            self.logger.info("✅ Kafka stream configured with explicit schema")
            return parsed_df
            
        except Exception as e:
            self.logger.error(f"Failed to read from Kafka: {e}")
            raise
    
    def convert_schema(self, df: DataFrame) -> DataFrame:
        """Convierte schema usando el conversor configurado."""
        try:
            df_converted = self.converter_function(df)
            return df_converted
        except Exception as e:
            self.logger.error(f"Schema conversion failed: {e}")
            raise
    
    def preprocess(self, df: DataFrame) -> DataFrame:
        """Aplica preprocesamiento usando función configurada."""
        try:
            df_processed = self.pipeline.transform(df)
            return self.pipeline.select_features(df_processed)
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
    
    def predict_batch(self, df_preprocessed: DataFrame) -> DataFrame:
        """Realiza predicciones sobre un batch."""
        try:
            # Configuración para la función standalone (solo primitivos)
            pred_config = {
                'url': str(self.ray_url),
                'batch_size': int(self.ray_batch_size),
                'timeout': int(self.ray_timeout),
                'id_column': str(self.id_column),
                'prediction_column': str(self.prediction_column),
                'payload_format': str(self.ray_payload_format),
                'prediction_key': str(self.ray_pred_key),
            }
            
            predictions_rdd = df_preprocessed.rdd.mapPartitions(
                lambda p: _predict_partition_ray(p, pred_config)
            )
            
            predictions_df = self.spark.createDataFrame(
                predictions_rdd, 
                schema=self.output_schema
            )
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def process_batch(self, batch_df: DataFrame, batch_id: int):
        """
        Procesa un micro-batch completo.
        Pipeline: convert → preprocess → predict → join → write
        
        Includes Prometheus metrics for latency tracking.
        """
        batch_start_time = time.time()
        
        try:
            self.logger.info(f"Processing batch {batch_id}")
            
            # Count records in batch
            record_count = batch_df.count()
            BATCH_RECORDS_TOTAL.set(record_count)
            
            if record_count == 0:
                self.logger.info(f"⏭️ Batch {batch_id} is empty, skipping")
                return
            
            # 1. Convertir schema de Kafka a features
            df_features = self.convert_schema(batch_df)
            
            # 2. Preprocesar (with latency tracking)
            preprocess_start = time.time()
            df_preprocessed = self.preprocess(df_features)
            preprocess_latency = time.time() - preprocess_start
            LATENCY_PREPROCESS.set(preprocess_latency)
            
            # 3. Predecir (with latency tracking)
            inference_start = time.time()
            predictions_df = self.predict_batch(df_preprocessed)
            inference_latency = time.time() - inference_start
            LATENCY_INFERENCE.set(inference_latency)
            INFERENCE_LATENCY_SUMMARY.observe(inference_latency)

            # 3.1 Distribution of predicted classes (for attack monitoring)
            try:
                # Force a compact aggregation: at most 6 rows.
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
            output_df = batch_df.join(
                predictions_df,
                on=self.id_column,
                how='inner'
            )
            
            # 5. Escribir a Kafka
            self._write_to_kafka(output_df)
            
            # Total batch latency (roundtrip)
            total_latency = time.time() - batch_start_time
            LATENCY_TOTAL_BATCH.set(total_latency)
            
            self.logger.info(
                f"✅ Batch {batch_id} completed | Records: {record_count} | "
                f"Preprocess: {preprocess_latency:.3f}s | Inference: {inference_latency:.3f}s | "
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
        Ejecuta el pipeline de inferencia.
        Pipeline completo: Kafka → Parse → Convert → Preprocess → Predict → Write
        """
        try:
            self.logger.info("🚀 Starting Generic Kafka-Spark Inference Pipeline")
            
            # Leer stream de Kafka
            df_stream = self.read_from_kafka()
            
            # Configurar streaming query con foreachBatch
            query = (
                df_stream
                .writeStream
                .foreachBatch(self.process_batch)
                .option("checkpointLocation", self.checkpoint_location)
                .outputMode("append")
                .trigger(processingTime=self.processing_time)
                .start()
            )
            
            self.logger.info(f"✅ Streaming query started")
            self.logger.info(f"📍 Checkpoint: {self.checkpoint_location}")
            self.logger.info(f"📥 Input topic: {self.kafka_input_topic}")
            self.logger.info(f"📤 Output topic: {self.kafka_output_topic}")
            self.logger.info(f"⚙️  Schema: {self.input_schema_name} → {self.output_schema_name}")
            self.logger.info(f"🔄 Converter: {self.converter_name}")
            self.logger.info(f"🧮 Preprocessing using Spark Custom Pipeline")
            self.logger.info("⏳ Waiting for termination...")
            
            # Esperar terminación
            query.awaitTermination()
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise

# ===== ENTRY POINT =====
def main():
    """Punto de entrada de la aplicación."""
    params_path = '/app/repo/k3s/params.yaml'
    pipeline = KafkaSparkInference(params_path=params_path)
    pipeline.run_inference()

if __name__ == "__main__":
    main()