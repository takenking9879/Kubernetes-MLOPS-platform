from prometheus_client import Counter, Gauge, Histogram, Summary

PREPROCESS_RECORDS = Counter('preprocess_records_total', 'Total records preprocessed', ['dataset'])
PREPROCESS_BATCHES = Counter('preprocess_batches_total', 'Total batches preprocessed', ['dataset'])
PREPROCESS_RECORDS_BY_CLASS = Counter('preprocess_records_by_class_total', 'Records per class', ['dataset', 'class'])
PREPROCESS_RECORDS_LAST_BATCH = Gauge('preprocess_records_last_batch', 'Records in last batch by class', ['dataset', 'class'])
PREPROCESS_BATCH_LATENCY = Histogram('preprocess_batch_latency_seconds', 'Batch preprocessing latency', ['dataset'], buckets=[1, 5, 10, 30, 60, 120, 300])
PREPROCESS_BATCH_LATENCY_LAST = Gauge('preprocess_batch_latency_last_seconds', 'Last dataset preprocessing latency (seconds)', ['dataset'])
PREPROCESS_ERRORS = Counter('preprocess_errors_total', 'Total preprocessing errors', ['dataset', 'error_type'])
PREPROCESS_CURRENT_DATASET = Gauge('preprocess_current_dataset_progress', 'Current dataset being processed (0=idle, 1=train, 2=val, 3=test)')

# Inference

# Custom metrics for MLOps observability
LATENCY_PREPROCESS = Gauge('latency_preprocess_seconds', 'Time spent in Spark preprocessing per batch')
LATENCY_INFERENCE = Gauge('latency_inference_seconds', 'Time spent calling Ray Serve per batch')
LATENCY_TOTAL_BATCH = Gauge('latency_total_batch_seconds', 'Total roundtrip time Kafka->Spark->Ray->Kafka per batch')
BATCH_RECORDS_TOTAL = Gauge('spark_batch_records_total', 'Number of records processed in the last batch')
BATCH_ERRORS_TOTAL = Counter('spark_batch_errors_total', 'Total number of batch processing errors')
INFERENCE_LATENCY_SUMMARY = Summary('inference_latency_summary_seconds', 'Summary of inference latencies')

# Predictions distribution (attack classes)
PREDICTIONS_BY_CLASS_TOTAL = Counter(
    'spark_predictions_by_class_total',
    'Total number of predictions produced, labeled by class',
    ['class'],
)
PREDICTIONS_BY_CLASS_LAST_BATCH = Gauge(
    'spark_predictions_by_class_last_batch',
    'Number of predictions in the most recent micro-batch, labeled by class',
    ['class'],
)

