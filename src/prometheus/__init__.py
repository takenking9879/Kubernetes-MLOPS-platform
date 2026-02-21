"""Prometheus metrics and callbacks for ML platform observability."""

from .metrics import (
    # Driver-side metrics (used by main.py orchestrator)
    TRAIN_FAILURES,
    TRAIN_SPLIT_ROWS,
    # Tune metrics
    TUNE_TRIALS,
    TUNE_TRIAL_STATUS,
    TUNE_BEST_METRIC,
    TUNE_TRIALS_BY_STATUS,
    # Worker-side registries and metrics
    create_worker_registry,
    # Helper functions
    export_final_metrics,
)
from .tune import PrometheusTuneCallback

__all__ = [
    # Driver metrics
    "TRAIN_FAILURES",
    "TRAIN_SPLIT_ROWS",
    # Tune
    "TUNE_TRIALS",
    "TUNE_TRIAL_STATUS",
    "TUNE_BEST_METRIC",
    "TUNE_TRIALS_BY_STATUS",
    "PrometheusTuneCallback",
    # Worker utilities
    "create_worker_registry",
    # Helpers
    "export_final_metrics",
]

from .preprocessing import (
    PREPROCESS_RECORDS,
    PREPROCESS_BATCHES,
    PREPROCESS_RECORDS_BY_CLASS,
    PREPROCESS_RECORDS_LAST_BATCH,
    PREPROCESS_BATCH_LATENCY,
    PREPROCESS_BATCH_LATENCY_LAST,
    PREPROCESS_ERRORS,
    PREPROCESS_CURRENT_DATASET,
    LATENCY_PREPROCESS,
    LATENCY_INFERENCE,
    LATENCY_TOTAL_BATCH,
    LATENCY_KAFKA_WRITE,
    BATCH_RECORDS_TOTAL,
    BATCH_ERRORS_TOTAL,
    INFERENCE_LATENCY_SUMMARY,
    PREDICTIONS_BY_CLASS_TOTAL,
    PREDICTIONS_BY_CLASS_LAST_BATCH,
)
__all__.extend([
    "PREPROCESS_RECORDS",
    "PREPROCESS_BATCHES",
    "PREPROCESS_RECORDS_BY_CLASS",
    "PREPROCESS_RECORDS_LAST_BATCH",
    "PREPROCESS_BATCH_LATENCY",
    "PREPROCESS_BATCH_LATENCY_LAST",
    "PREPROCESS_ERRORS",
    "PREPROCESS_CURRENT_DATASET",
    "LATENCY_PREPROCESS",
    "LATENCY_INFERENCE",
    "LATENCY_TOTAL_BATCH",
    "LATENCY_KAFKA_WRITE",
    "BATCH_RECORDS_TOTAL",
    "BATCH_ERRORS_TOTAL",
    "INFERENCE_LATENCY_SUMMARY",
    "PREDICTIONS_BY_CLASS_TOTAL",
    "PREDICTIONS_BY_CLASS_LAST_BATCH",
])