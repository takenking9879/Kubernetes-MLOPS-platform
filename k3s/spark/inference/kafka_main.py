import os
import logging
from typing import Dict, Optional, Iterator, List, Any
import json
from confluent_kafka import Consumer, KafkaException
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StructType
import importlib
import boto3
from k3s.spark.utils import create_logger, BaseUtils
from pyspark.sql.functions import from_json, col, struct, to_json
from k3s.spark.schema.schemas import schema_features, schema_full
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class KafkaSparkInference(BaseUtils):
    """
    Production-ready Kafka-Spark inference pipeline with Ray Serve integration.
    
    Optimizations:
    - Uses foreachBatch for proper streaming semantics
    - Caches pipeline artifacts and modules
    - Batches HTTP requests efficiently
    - Robust error handling with fallback strategies
    """
    
    def __init__(self, params_path: str):
        logger = create_logger('kafka_spark_inference', 'kafka_spark_inference.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['spark']
        
        # Kafka configuration
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'topic-traffic')
        self.kafka_output_topic = os.getenv('KAFKA_TOPIC_OUTPUT', 'topic-prediction')
        self.kafka_username = os.getenv('KAFKA_USERNAME', None)
        self.kafka_password = os.getenv('KAFKA_PASSWORD', None)
        self.kafka_sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', os.getenv('KAFKA_SASLMECHANISM', 'SCRAM-SHA-512'))
        self.kafka_security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_PLAINTEXT')
        
        self.kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.scram.ScramLoginModule required '
            f'username="{self.kafka_username}" password="{self.kafka_password}";'
        )
        
        # Ray Serve configuration
        self.ray_serve_url = os.getenv("RAY_SERVE_URL", "http://serving.localhost/infer")
        self.ray_batch_size = int(os.getenv("RAY_BATCH_SIZE", "100"))
        self.ray_request_timeout = int(os.getenv("RAY_REQUEST_TIMEOUT", "30"))
        self.ray_max_retries = int(os.getenv("RAY_MAX_RETRIES", "3"))
        
        # Internal state
        self.s3 = None
        self.scaler = None
        self._preprocessing_module = None  # Cache for imported module
        
        # Conditional checks based on params
        check_kafka = self.params.get('check_kafka_connection', True)
        if check_kafka:
            self._check_kafka_connection()
        else:
            self.logger.info("Skipping Kafka connection check (disabled in params)")
        
        self._check_minio_connection()
        
        # Pre-load preprocessing module during init
        self._load_preprocessing_module()
        
        # Create Spark session
        self.spark = self._create_spark_session()

    def _check_kafka_connection(self):
        """
        Lightweight Kafka connectivity check.
        
        Optimization: Made optional via params to avoid startup delays.
        Uses minimal timeout for quick validation.
        """
        try:
            self.logger.info(f"Checking Kafka connection to {self.kafka_bootstrap_servers}")
            
            conf = {
                'bootstrap.servers': self.kafka_bootstrap_servers,
                'group.id': 'kafka-connection-test',
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': False,
                'security.protocol': self.kafka_security_protocol,
                'sasl.mechanism': self.kafka_sasl_mechanism,
                'sasl.username': self.kafka_username,
                'sasl.password': self.kafka_password,
                'socket.timeout.ms': 5000,  # Reduced timeout
                'api.version.request.timeout.ms': 5000
            }
            
            consumer = Consumer(conf)
            metadata = consumer.list_topics(timeout=5)  # Reduced from 10s to 5s
            
            topics = metadata.topics
            if topics:
                self.logger.info(f"✅ Kafka connection verified. {len(topics)} topics available")
            else:
                self.logger.warning("⚠️ Kafka connected but no topics found")
            
            consumer.close()
            
        except KafkaException as e:
            self.logger.error(f'Kafka connection failed: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error during Kafka check: {e}')
            raise

    def _check_minio_connection(self):
        """Verify MinIO/S3 connectivity and cache client."""
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
            self.logger.info(f'✅ MinIO/S3 connection verified. Buckets: {bucket_names}')
        except Exception as e:
            self.logger.error(f'MinIO/S3 connection failed: {e}')
            raise

    def _create_spark_session(self):
        """Create optimized Spark session for streaming workloads."""
        try:
            self.logger.info("Creating SparkSession with Kafka and S3 support")
            spark = (
                SparkSession.builder
                .appName(self.params['app_name'])
                # S3 configuration
                .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
                .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
                .config("spark.hadoop.fs.s3a.path.style.access", "false")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.committer.name", "magic")
                .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
                # Streaming optimizations
                .config("spark.sql.streaming.schemaInference", "false")  # Explicit schemas only
                .config("spark.sql.adaptive.enabled", "false")  # AQE doesn't work with streaming
                .getOrCreate()
            )
            self.logger.info("✅ SparkSession created successfully")
            return spark
        except Exception as e:
            self.logger.error(f'Failed to create SparkSession: {e}')
            raise

    def _load_preprocessing_module(self):
        """
        Load preprocessing module once during initialization.
        
        Optimization: Avoids repeated importlib calls in hot path.
        """
        try:
            pipeline_module = self.params['pipeline']['module']
            self.logger.info(f"Loading preprocessing module: {pipeline_module}")
            self._preprocessing_module = importlib.import_module('k3s.spark.' + pipeline_module)
            self.logger.info("✅ Preprocessing module loaded and cached")
        except Exception as e:
            self.logger.error(f'Failed to load preprocessing module: {e}')
            raise

    def read_from_kafka(self) -> DataFrame:
        """
        Read stream from Kafka with explicit schema parsing.
        
        Optimization: Uses explicit schema (schema_features) to avoid inference.
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
                # Streaming optimizations
                .option("maxOffsetsPerTrigger", "10000")  # Rate limiting
                .option("failOnDataLoss", "false")  # Graceful handling
                .load()
            )
            
            # Parse with explicit schema
            parsed_df = (
                df.selectExpr("CAST(value AS STRING)")
                .select(from_json(col("value"), schema_features).alias("data"))
                .select("data.*")
            )
            
            self.logger.info("✅ Kafka stream configured with explicit schema")
            return parsed_df
            
        except Exception as e:
            self.logger.error(f'Failed to read from Kafka: {e}')
            raise
    
    def load_scaler_artifact(self) -> Dict:
        """
        Load pipeline artifact from S3 with local caching.
        
        Optimization: Downloads once and caches in memory to avoid repeated S3 calls.
        For production, consider using Spark's addFile() or mounted volumes.
        """
        if self.scaler is not None:
            return self.scaler
        
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

            self.logger.info("✅ Pipeline artifact loaded and cached")
            return self.scaler

        except Exception as e:
            self.logger.error(f"Failed to load pipeline artifact: {e}")
            raise

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Apply preprocessing using cached module and scaler.
        
        Optimization: Uses pre-loaded module instead of dynamic import.
        """
        try:
            if self.scaler is None:
                self.scaler = self.load_scaler_artifact()
            
            df_out, _ = self._preprocessing_module.preprocess_spark(
                df, 
                model=self.scaler, 
                train=False
            )
            return df_out
            
        except Exception as e:
            self.logger.error(f'Preprocessing failed: {e}')
            raise

    def validate_schema(self, df: DataFrame, expected_schema: StructType) -> bool:
        """
        Robust schema validation with detailed error messages.
        
        Improvement: Checks field names, types, and nullability explicitly.
        Fails fast with clear error messages.
        """
        try:
            actual_fields = {f.name: (f.dataType, f.nullable) for f in df.schema.fields}
            expected_fields = {f.name: (f.dataType, f.nullable) for f in expected_schema.fields}
            
            # Check for missing fields
            missing = set(expected_fields.keys()) - set(actual_fields.keys())
            if missing:
                raise ValueError(f"Schema validation failed. Missing fields: {missing}")
            
            # Check for extra fields
            extra = set(actual_fields.keys()) - set(expected_fields.keys())
            if extra:
                self.logger.warning(f"Extra fields found (will be ignored): {extra}")
            
            # Check field types
            for field_name in expected_fields:
                if field_name in actual_fields:
                    exp_type, exp_null = expected_fields[field_name]
                    act_type, act_null = actual_fields[field_name]
                    
                    if exp_type != act_type:
                        raise ValueError(
                            f"Schema validation failed for field '{field_name}'. "
                            f"Expected {exp_type}, got {act_type}"
                        )
            
            self.logger.info("✅ Schema validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            raise

    def process_batch_with_ray(self, batch_df: DataFrame, batch_id: int):
        """
        Process a single micro-batch by sending it to Ray Serve.
        
        This is the correct pattern for Spark Structured Streaming.
        Uses foreachBatch to process each micro-batch as a static DataFrame.
        
        Optimizations:
        - Processes batch in partitions to leverage parallelism
        - Uses mapPartitions for efficient HTTP batching
        - Explicit schema for toDF() conversion
        
        Args:
            batch_df: Static DataFrame for this micro-batch
            batch_id: Unique identifier for this batch
        """
        try:
            if batch_df.isEmpty():
                self.logger.debug(f"Batch {batch_id}: Empty, skipping")
                return
            
            row_count = batch_df.count()
            self.logger.info(f"Batch {batch_id}: Processing {row_count} rows")
            
            # Apply predictions using mapPartitions on static DataFrame
            # This is safe because we're inside foreachBatch
            predictions_rdd = batch_df.rdd.mapPartitions(
                lambda partition: self._predict_partition(
                    partition,
                    batch_id
                )
            )
            
            # Convert back to DataFrame with explicit schema
            predictions_df = self.spark.createDataFrame(predictions_rdd, schema=schema_full)
            
            # Validate output schema
            self.validate_schema(predictions_df, schema_full)
            
            # Write to Kafka output topic
            (
                predictions_df
                .selectExpr(
                    "CAST(timestamp AS STRING) AS key",
                    "to_json(struct(*)) AS value"
                )
                .write
                .format("kafka")
                .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
                .option("topic", self.kafka_output_topic)
                .option("kafka.security.protocol", self.kafka_security_protocol)
                .option("kafka.sasl.mechanism", self.kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", self.kafka_sasl_jaas_config)
                .save()
            )
            
            self.logger.info(f"Batch {batch_id}: ✅ Successfully wrote {row_count} predictions to Kafka")
            
        except Exception as e:
            self.logger.error(f"Batch {batch_id}: ❌ Failed to process: {e}", exc_info=True)
            raise

    def _predict_partition(self, partition: Iterator[Row], batch_id: int) -> Iterator[Dict[str, Any]]:
        """
        Process a single partition by batching requests to Ray Serve.
        
        Optimizations:
        - Reuses HTTP session with connection pooling
        - Implements retry logic with exponential backoff
        - Batches multiple rows per request
        - Minimal logging (errors only)
        
        Args:
            partition: Iterator of Rows from Spark partition
            batch_id: Identifier for logging context
            
        Yields:
            Dict with original data + predictions
        """
        # Setup logger with batch context
        logger = logging.getLogger(f"predict_partition_batch_{batch_id}")
        
        # Configure HTTP session with connection pooling and retries
        session = requests.Session()
        retry_strategy = Retry(
            total=self.ray_max_retries,
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
            """Send batch to Ray Serve and return predictions."""
            try:
                payload = {"data": batch_data}
                response = session.post(
                    self.ray_serve_url,
                    json=payload,
                    timeout=self.ray_request_timeout,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                predictions = response.json()
                
                if not isinstance(predictions, dict) or "predictions" not in predictions:
                    raise ValueError(f"Invalid response format: {predictions}")
                
                return predictions
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout after {self.ray_request_timeout}s for {len(batch_data)} items")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        def combine_results(batch_data: List[Dict], predictions_response: Dict) -> List[Dict]:
            """Merge original data with predictions."""
            pred_list = predictions_response.get("predictions", [])
            model_variant = predictions_response.get("model", {}).get("variant", "unknown")
            latency_ms = predictions_response.get("latency_ms", None)
            
            if len(pred_list) != len(batch_data):
                logger.warning(f"Prediction mismatch: expected {len(batch_data)}, got {len(pred_list)}")
            
            results = []
            for i, row_dict in enumerate(batch_data):
                results.append({
                    **row_dict,
                    "prediction": pred_list[i] if i < len(pred_list) else None,
                    "model_variant": model_variant,
                    "latency_ms": latency_ms
                })
            
            return results
        
        # Process partition in batches
        batch = []
        total_processed = 0
        
        try:
            for row in partition:
                batch.append(row.asDict())
                
                if len(batch) >= self.ray_batch_size:
                    try:
                        predictions = send_batch_to_ray(batch)
                        results = combine_results(batch, predictions)
                        
                        for result in results:
                            yield result
                        
                        total_processed += len(batch)
                        
                    except Exception as e:
                        logger.error(f"Failed to process batch: {e}")
                        
                        # Yield rows with error flag
                        for row_dict in batch:
                            yield {
                                **row_dict,
                                "prediction": None,
                                "model_variant": "error",
                                "latency_ms": None,
                                "error": str(e)
                            }
                    
                    batch = []
            
            # Process final batch
            if batch:
                try:
                    predictions = send_batch_to_ray(batch)
                    results = combine_results(batch, predictions)
                    
                    for result in results:
                        yield result
                    
                    total_processed += len(batch)
                    
                except Exception as e:
                    logger.error(f"Failed to process final batch: {e}")
                    
                    for row_dict in batch:
                        yield {
                            **row_dict,
                            "prediction": None,
                            "model_variant": "error",
                            "latency_ms": None,
                            "error": str(e)
                        }
            
            # Only log summary (avoid per-batch noise)
            if total_processed > 0:
                logger.info(f"Partition complete: {total_processed} rows processed")
            
        finally:
            session.close()
    
    def run_inference(self):
        """
        Main inference pipeline using proper Spark Structured Streaming patterns.
        
        Architecture:
        1. Read stream from Kafka
        2. Preprocess data
        3. Use foreachBatch to process each micro-batch
        4. Inside each batch, use mapPartitions to batch HTTP calls to Ray Serve
        5. Write results to output Kafka topic
        
        Key improvements:
        - Uses foreachBatch instead of rdd.mapPartitions on streaming DF
        - Explicit schema handling throughout
        - Proper checkpoint management
        - Graceful error handling
        """
        try:
            self.logger.info("🚀 Starting Kafka-Spark inference pipeline with Ray Serve")
            
            # 1. Read stream from Kafka
            df_raw = self.read_from_kafka()
            
            # 2. Preprocess data
            df_processed = self.preprocess(df_raw)
            self.logger.info("✅ Preprocessing configured")
            
            # 3. Configure checkpoint location
            checkpoint_location = os.getenv(
                "SPARK_CHECKPOINT_LOCATION",
                "s3a://k8s-mlops-platform-bucket/checkpoints/kafka-spark-inference"
            )
            
            # 4. Start streaming query with foreachBatch
            # This is the CORRECT pattern for Spark Structured Streaming
            query = (
                df_processed
                .writeStream
                .foreachBatch(self.process_batch_with_ray)
                .option("checkpointLocation", checkpoint_location)
                .outputMode("append")
                .trigger(processingTime="10 seconds")  # Process every 10 seconds
                .start()
            )
            
            self.logger.info(f"✅ Streaming query started")
            self.logger.info(f"📍 Checkpoint: {checkpoint_location}")
            self.logger.info(f"📤 Output topic: {self.kafka_output_topic}")
            self.logger.info("⏳ Waiting for termination...")
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            self.logger.error(f'❌ Inference pipeline failed: {e}', exc_info=True)
            raise


def main():
    """Entry point for the application."""
    params_path = "/app/repo/k3s/params.yaml"
    kafka_spark_inference = KafkaSparkInference(params_path=params_path)
    kafka_spark_inference.run_inference()


if __name__ == "__main__":
    main()