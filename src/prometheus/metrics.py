"""Centralized Prometheus metric definitions.

This module provides:
- Driver-side metrics (used by main.py orchestrator)
- Worker-side metric factory for distributed training
"""

from prometheus_client import Counter, Gauge, CollectorRegistry

# ----------------------
# Driver-side metrics (default registry)
# ----------------------
# These are used by main.py to track orchestration-level metrics.

# Training orchestration
TRAIN_FAILURES = Counter("train_failures_total", "Total training failures", ["framework", "error_type"])
TRAIN_SPLIT_ROWS = Gauge("train_split_rows", "Dataset rows per split", ["framework", "split"])

# Final training metrics (for gauge panels after training completes)
TRAIN_LOSS = Gauge("train_loss", "Training loss", ["framework", "split"])
TRAIN_ACCURACY = Gauge("train_accuracy", "Training accuracy", ["framework", "split"])
TRAIN_F1 = Gauge("train_f1", "Training F1 score", ["framework", "split"])
TRAIN_PRECISION = Gauge("train_precision", "Training precision", ["framework", "split"])
TRAIN_RECALL = Gauge("train_recall", "Training recall", ["framework", "split"])

# Tuning orchestration
TUNE_TRIALS = Counter("tune_trials_total", "Total tuning trials", ["framework"])
TUNE_TRIAL_STATUS = Gauge(
    "tune_trial_status",
    "Trial status (1=running, 2=success, 3=failed)",
    ["framework", "trial_id"],
)
TUNE_BEST_METRIC = Gauge(
    "tune_best_metric_value",
    "Best metric value found",
    ["framework", "metric"],
)
TUNE_TRIALS_BY_STATUS = Gauge(
    "tune_trials_by_status",
    "Number of trials by status",
    ["framework", "status"],
)


# ----------------------
# Helper functions
# ----------------------
def export_final_metrics(framework: str, metrics: dict) -> None:
    """Export final training metrics to default Prometheus registry.
    
    Note: PushGateway is deprecated. We rely on worker HTTP servers + grace period
    for metric persistence during pod termination.
    
    Args:
        framework: "pytorch" or "xgboost"
        metrics: Dict with training metrics (val_loss, val_accuracy, etc.)
    """
    if not metrics:
        return
    
    try:
        # Set metrics in default registry (for immediate scraping if pod alive)
        # PyTorch metrics
        if "val_loss" in metrics:
            TRAIN_LOSS.labels(framework=framework, split="val").set(float(metrics["val_loss"]))
        if "val_accuracy" in metrics:
            TRAIN_ACCURACY.labels(framework=framework, split="val").set(float(metrics["val_accuracy"]))
        if "val_f1_avg" in metrics:
            TRAIN_F1.labels(framework=framework, split="val").set(float(metrics["val_f1_avg"]))
        if "val_precision_avg" in metrics:
            TRAIN_PRECISION.labels(framework=framework, split="val").set(float(metrics["val_precision_avg"]))
        if "val_recall_avg" in metrics:
            TRAIN_RECALL.labels(framework=framework, split="val").set(float(metrics["val_recall_avg"]))
        
        # XGBoost metrics (convert to common format)
        if "validation-mlogloss" in metrics:
            TRAIN_LOSS.labels(framework=framework, split="val").set(float(metrics["validation-mlogloss"]))
            
    except Exception as e:
        print(f"[prometheus] Failed to export final metrics: {e}")


# ----------------------
# Worker-side metric factory
# ----------------------
def create_worker_registry(framework: str) -> tuple[CollectorRegistry, dict]:
    """Create isolated worker registry with training metrics.
    
    Returns:
        (registry, metrics_dict) where metrics_dict contains:
        - current_epoch: Gauge for current epoch/iteration
        - epoch_duration: Gauge for last epoch duration
        - loss: Gauge with labels [split] for train/val loss
        - accuracy: Gauge with labels [split] (optional, for classification)
        - f1: Gauge with labels [split] (optional, for multiclass)
        - precision: Gauge with labels [split] (optional)
        - recall: Gauge with labels [split] (optional)
    """
    registry = CollectorRegistry(auto_describe=True)
    
    metrics = {
        "current_epoch": Gauge(
            "train_current_epoch",
            "Current epoch/iteration number",
            ["framework"],
            registry=registry
        ),
        "epoch_duration": Gauge(
            "train_epoch_duration_last_seconds",
            "Last reported iteration duration (seconds)",
            ["framework"],
            registry=registry
        ),
        "loss": Gauge(
            "train_loss",
            "Current training loss",
            ["framework", "split"],
            registry=registry
        ),
        "accuracy": Gauge(
            "train_accuracy",
            "Current accuracy metric",
            ["framework", "split"],
            registry=registry
        ),
        "f1": Gauge(
            "train_f1",
            "Current F1 score",
            ["framework", "split"],
            registry=registry
        ),
        "precision": Gauge(
            "train_precision",
            "Current precision",
            ["framework", "split"],
            registry=registry
        ),
        "recall": Gauge(
            "train_recall",
            "Current recall",
            ["framework", "split"],
            registry=registry
        ),
    }
    
    return registry, metrics
