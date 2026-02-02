import os
import logging
from typing import Dict, Optional, Iterator, List, Any
import json
import ray
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
from pyspark.sql import SparkSession, Row
import importlib
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType
import boto3
from k3s.spark.utils import create_logger, BaseUtils
from pyspark.sql.functions import (from_json, col)
from schema.schemas import schema_features
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class KafkaSparkInference(BaseUtils):
    def __init__(self, params_path: str):
        logger = create_logger('kafka_spark_inference', 'kafka_spark_inference.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['spark']
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'topic-traffic')
        self.kafka_output_topic = os.getenv('KAFKA_TOPIC_OUTPUT', 'topic-prediction')
        self.kafka_username = os.getenv('KAFKA_USERNAME', None)
        self.kafka_password = os.getenv('KAFKA_PASSWORD', None)
        self.kafka_sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', os.getenv('KAFKA_SASLMECHANISM', 'SCRAM-SHA-512'))
        self.kafka_security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_PLAINTEXT')
        self.s3 = None
        self.scaler = None
    
        self.kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.scram.ScramLoginModule required '
            f'username="{self.kafka_username}" password="{self.kafka_password}";'
        )

        self._check_kafka_connection()
        self._check_minio_connection()

        self.spark = self._create_spark_session()

    def _check_kafka_connection(self):
        """Verifica la conectividad con Kafka usando confluent_kafka"""
        try:
            self.logger.info(f"Checking Kafka connection to {self.kafka_bootstrap_servers}")
            
            # Configuración base para el consumer de prueba
            conf = {
                'bootstrap.servers': self.kafka_bootstrap_servers,
                'group.id': 'kafka-connection-test',
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': False,
                'security.protocol': self.kafka_security_protocol,
                'sasl.mechanism': self.kafka_sasl_mechanism,
                'sasl.username': self.kafka_username,
                'sasl.password': self.kafka_password,
            }
            
            # Crear consumer temporal para verificar conexión
            consumer = Consumer(conf)
            
            # Obtener metadata del cluster (esto fuerza la conexión)
            metadata = consumer.list_topics(timeout=10)
            
            # Verificar que el topic existe
            topics = metadata.topics
            if topics:
                self.logger.info(f"✅ Kafka connection verified. Topics '{topics}' found.")
                self.logger.info(f"Available topics: {list(topics.keys())}")
            else:
                self.logger.warning("⚠️ Kafka connection verified but no topics found.")
            # Cerrar consumer
            consumer.close()            
        except KafkaException as e:
            self.logger.error(f'Kafka connection failed: {e}', exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error during Kafka connection test: {e}', exc_info=True)
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

    def _create_spark_session(self):
        try:
            self.logger.info("Creating SparkSession with Kafka support")
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
            # 2. Magic Committer (Ya lo tenías, ¡excelente!)
            .config("spark.hadoop.fs.s3a.committer.name", "magic")
            .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
            .getOrCreate()
            )
            self.logger.info("SparkSession created successfully")
            return spark
        except Exception as e:
            self.logger.error('Failed to create SparkSession: %s', str(e), exc_info=True)
            raise

    def read_from_kafka(self):
        """
        Lee mensajes desde un tópico de Kafka usando Spark Structured Streaming.
        """
        try:
            self.logger.info(f"Connecting to Kafka topic: {self.kafka_topic}")
            df = (
                self.spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
                .option("subscribe", self.kafka_topic)
                .option("startingOffsets", "latest")
                .option("kafka.security.protocol", self.kafka_security_protocol)
                .option("kafka.sasl.mechanism", self.kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", self.kafka_sasl_jaas_config)
                .load()
            )
            from k3s.spark.schema.schemas import schema_features
            parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema_features).alias("data")) \
            .select("data.*")
            self.logger.info("Successfully connected to Kafka and read stream")
            return parsed_df
        except Exception as e:
            self.logger.error('Failed to read from Kafka: %s', str(e), exc_info=True)
            raise
    
    def load_scaler_artifact(self):
        """Carga el pipeline primero desde local; si no existe, lo baja de S3."""
        try:
            self.logger.info("Loading pipeline artifact from S3")
            local_path = "/tmp/pipeline_model.json"
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

    def preprocess(self, df):
        """
        Unified preprocessing entry.
        """
        try:
            pipeline_module = self.params['pipeline']['module']
            self.logger.info(f"Loading feature pipeline: {pipeline_module}")
            module = importlib.import_module(pipeline_module)
            if self.scaler is None:
                self.scaler = self.load_scaler_artifact()
            df_out, _ = module.preprocess_spark(df, model=self.scaler, train=False)
            return df_out
        except Exception as e:
            self.logger.error('Preprocess failed to complete: %s', str(e), exc_info=True)
            raise
    
    @staticmethod
    def predict_with_ray_serve(partition: Iterator[Row]) -> Iterator[Dict[str, Any]]:
        """
        Envía batches de datos a Ray Serve y retorna predicciones.
        
        Optimizaciones:
        - Reutiliza sesión HTTP con connection pooling
        - Implementa reintentos automáticos
        - Manejo robusto de errores
        - Logging mejorado
        - Validación de respuestas
        
        Args:
            partition: Iterator de Rows de Spark
            
        Yields:
            Dict con datos originales + predicciones
        """
        # Configuración desde variables de entorno
        ray_serve_url = os.getenv("RAY_SERVE_URL", "http://serving.localhost/infer")
        batch_size = int(os.getenv("RAY_BATCH_SIZE", "100"))
        request_timeout = int(os.getenv("RAY_REQUEST_TIMEOUT", "30"))
        max_retries = int(os.getenv("RAY_MAX_RETRIES", "3"))
        
        # Logger para esta función
        logger = logging.getLogger("predict_with_ray_serve")
        
        # Configurar sesión HTTP con connection pooling y reintentos
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        def send_batch_to_ray(batch_data: List[Dict]) -> Dict:
            """Envía un batch al servicio Ray y retorna la respuesta"""
            try:
                payload = {"data": batch_data}
                response = session.post(
                    ray_serve_url, 
                    json=payload, 
                    timeout=request_timeout,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                predictions = response.json()
                
                # Validar estructura de respuesta
                if not isinstance(predictions, dict):
                    raise ValueError(f"Invalid response format: expected dict, got {type(predictions)}")
                
                if "predictions" not in predictions:
                    raise ValueError("Response missing 'predictions' field")
                
                return predictions
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout after {request_timeout}s for batch of {len(batch_data)} items")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error sending batch: {e}")
                raise
        
        def combine_results(batch_data: List[Dict], predictions_response: Dict) -> List[Dict]:
            """Combina datos originales con predicciones"""
            pred_list = predictions_response.get("predictions", [])
            model_variant = predictions_response.get("model", {}).get("variant", "unknown")
            latency_ms = predictions_response.get("latency_ms", None)
            
            if len(pred_list) != len(batch_data):
                logger.warning(
                    f"Prediction count mismatch: expected {len(batch_data)}, got {len(pred_list)}"
                )
            
            results = []
            for i, row_dict in enumerate(batch_data):
                result = {
                    **row_dict,
                    "prediction": pred_list[i] if i < len(pred_list) else None,
                    "model_variant": model_variant,
                    "latency_ms": latency_ms
                }
                results.append(result)
            
            return results
        
        # Procesar partición en batches
        batch = []
        total_processed = 0
        total_errors = 0
        
        try:
            for row in partition:
                batch.append(row.asDict())
                
                # Enviar batch cuando alcanza el tamaño configurado
                if len(batch) >= batch_size:
                    try:
                        predictions = send_batch_to_ray(batch)
                        results = combine_results(batch, predictions)
                        
                        for result in results:
                            yield result
                        
                        total_processed += len(batch)
                        logger.info(f"Successfully processed batch of {len(batch)} items. Total: {total_processed}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process batch: {e}")
                        total_errors += len(batch)
                        
                        # Yield resultados con error flag
                        for row_dict in batch:
                            yield {
                                **row_dict,
                                "prediction": None,
                                "model_variant": "error",
                                "latency_ms": None,
                                "error": str(e)
                            }
                    
                    batch = []
            
            # Procesar batch final (si existe)
            if batch:
                try:
                    predictions = send_batch_to_ray(batch)
                    results = combine_results(batch, predictions)
                    
                    for result in results:
                        yield result
                    
                    total_processed += len(batch)
                    logger.info(f"Successfully processed final batch of {len(batch)} items. Total: {total_processed}")
                    
                except Exception as e:
                    logger.error(f"Failed to process final batch: {e}")
                    total_errors += len(batch)
                    
                    for row_dict in batch:
                        yield {
                            **row_dict,
                            "prediction": None,
                            "model_variant": "error",
                            "latency_ms": None,
                            "error": str(e)
                        }
            
            logger.info(f"Partition complete. Processed: {total_processed}, Errors: {total_errors}")
            
        finally:
            # Cerrar sesión HTTP
            session.close()
    
    def run_inference(self):
        """
        Pipeline de inferencia Kafka-Spark con Ray Serve.
        
        Flujo:
        1. Lee stream desde Kafka
        2. Preprocesa datos
        3. Envía a Ray Serve para predicciones (en batches por partición)
        4. Escribe resultados a Kafka de salida
        """
        try:
            self.logger.info("Starting Kafka-Spark inference pipeline with Ray Serve")
            
            # 1. Leer stream desde Kafka
            df_raw = self.read_from_kafka()
            self.logger.info("Successfully connected to Kafka input topic")
            
            # 2. Preprocesar datos
            df_processed = self.preprocess(df_raw)
            self.logger.info("Data preprocessing configured")
            
            # 3. Aplicar predicciones usando Ray Serve
            # Nota: predict_with_ray_serve ahora es un método estático
            df_predictions = df_processed.rdd.mapPartitions(
                KafkaSparkInference.predict_with_ray_serve
            ).toDF()
            
            self.logger.info("Prediction pipeline configured")
            
            # 4. Escribir resultados a Kafka de salida
            checkpoint_location = os.getenv(
                "SPARK_CHECKPOINT_LOCATION",
                "s3a://k8s-mlops-platform-bucket/checkpoints/"
            )
            
            query = (
                df_predictions
                .selectExpr(
                    "CAST(timestamp AS STRING) AS key",
                    "to_json(struct(*)) AS value"
                )
                .writeStream
                .format("kafka")
                .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
                .option("topic", self.kafka_output_topic)
                .option("kafka.security.protocol", self.kafka_security_protocol)
                .option("kafka.sasl.mechanism", self.kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", self.kafka_sasl_jaas_config)
                .option("checkpointLocation", checkpoint_location)
                .outputMode("append")
                .start()
            )
            
            self.logger.info(f"Writing predictions to Kafka topic: {self.kafka_output_topic}")
            self.logger.info(f"Checkpoint location: {checkpoint_location}")
            self.logger.info("Streaming query started. Waiting for termination...")
            
            # Esperar a que termine el stream
            query.awaitTermination()
            
        except Exception as e:
            self.logger.error('Inference pipeline failed: %s', str(e), exc_info=True)
            raise


def main():
    params_path = "/app/repo/k3s/params.yaml"
    kafka_spark_inference = KafkaSparkInference(params_path=params_path)
    kafka_spark_inference.run_inference()


if __name__ == "__main__":
    main()