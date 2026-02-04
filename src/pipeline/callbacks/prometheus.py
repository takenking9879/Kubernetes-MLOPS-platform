"""Prometheus metrics + Ray Train/Tune callbacks.

Defined in an importable module (not __main__) so Ray Tune can cloudpickle
callbacks without capturing non-picklable Prometheus internals (thread locks).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

# ----------------------
# Metrics (driver-side)
# ----------------------

# Training metrics
TRAIN_EPOCHS = Counter("train_epochs_total", "Total epochs completed", ["framework"])
TRAIN_SAMPLES = Counter("train_samples_total", "Total samples processed", ["framework", "split"])
TRAIN_LOSS = Gauge("train_loss", "Current training loss", ["framework", "split"])
TRAIN_ACCURACY = Gauge("train_accuracy", "Current accuracy metric", ["framework", "split"])
TRAIN_F1 = Gauge("train_f1", "Current F1 score", ["framework", "split"])
TRAIN_PRECISION = Gauge("train_precision", "Current precision", ["framework", "split"])
TRAIN_RECALL = Gauge("train_recall", "Current recall", ["framework", "split"])
TRAIN_EPOCH_DURATION = Histogram(
    "train_epoch_duration_seconds",
    "Duration per epoch",
    ["framework"],
    buckets=[10, 30, 60, 120, 300, 600],
)
TRAIN_FAILURES = Counter("train_failures_total", "Total training failures", ["framework", "error_type"])
TRAIN_CURRENT_EPOCH = Gauge("train_current_epoch", "Current epoch number", ["framework"])
TRAIN_SPLIT_ROWS = Gauge("train_split_rows", "Dataset rows per split", ["framework", "split"])
TRAIN_EPOCH_DURATION_LAST = Gauge(
    "train_epoch_duration_last_seconds",
    "Last reported epoch/iteration duration (seconds)",
    ["framework"],
)

# Tuning metrics
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
# Callback base classes
# ----------------------

try:
    # Ray 2.5x+ provides UserCallback.
    from ray.train import UserCallback as _TrainingCallback
except Exception:  # pragma: no cover
    try:
        from ray.train.callbacks import TrainingCallback as _TrainingCallback  # type: ignore
    except Exception:  # pragma: no cover
        _TrainingCallback = object  # type: ignore

try:
    from ray.tune import Callback as _TuneCallback
except Exception:  # pragma: no cover
    _TuneCallback = object  # type: ignore


class PrometheusTrainCallback(_TrainingCallback):
    """Updates Prometheus gauges/counters from Ray Train results."""

    def __init__(self, *, framework: str):
        self.framework = framework
        self._last_step = 0

    # Make instance state trivially picklable.
    def __getstate__(self) -> Dict[str, Any]:
        return {"framework": self.framework, "_last_step": self._last_step}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.framework = state.get("framework", "unknown")
        self._last_step = int(state.get("_last_step", 0) or 0)

    def handle_result(self, results: Any, **kwargs: Any) -> None:  # Ray Train hook
        metrics = results or {}
        if not isinstance(metrics, dict):
            return

        # Determine step (epoch or iteration)
        step: Optional[int] = None
        if "epoch" in metrics:
            try:
                step = int(metrics["epoch"]) + 1
            except Exception:
                step = None
        elif "training_iteration" in metrics:
            try:
                step = int(metrics["training_iteration"])
            except Exception:
                step = None

        if step is not None and step > 0:
            delta = step if self._last_step <= 0 else max(0, step - self._last_step)
            if delta:
                TRAIN_EPOCHS.labels(framework=self.framework).inc(delta)
            TRAIN_CURRENT_EPOCH.labels(framework=self.framework).set(step)
            self._last_step = max(self._last_step, step)

        # Duration signals (prefer explicit epoch_time_sec)
        duration: Optional[float] = None
        for k in ("epoch_time_sec", "time_this_iter_s"):
            if k in metrics:
                try:
                    duration = float(metrics[k])
                except Exception:
                    duration = None
                break
        if duration is not None and duration >= 0:
            TRAIN_EPOCH_DURATION.labels(framework=self.framework).observe(duration)
            TRAIN_EPOCH_DURATION_LAST.labels(framework=self.framework).set(duration)

        # Loss metrics
        if "train_loss" in metrics:
            try:
                TRAIN_LOSS.labels(framework=self.framework, split="train").set(float(metrics["train_loss"]))
            except Exception:
                pass
        if "val_loss" in metrics:
            try:
                TRAIN_LOSS.labels(framework=self.framework, split="val").set(float(metrics["val_loss"]))
            except Exception:
                pass

        # XGBoost periodic callback metrics
        if "validation-mlogloss" in metrics:
            try:
                TRAIN_LOSS.labels(framework=self.framework, split="val").set(float(metrics["validation-mlogloss"]))
            except Exception:
                pass
        if "validation-merror" in metrics:
            try:
                merror = float(metrics["validation-merror"])
                TRAIN_ACCURACY.labels(framework=self.framework, split="val").set(
                    max(0.0, min(1.0, 1.0 - merror))
                )
            except Exception:
                pass

        # Multiclass metrics emitted by PyTorch train loop
        if "val_accuracy" in metrics:
            try:
                TRAIN_ACCURACY.labels(framework=self.framework, split="val").set(float(metrics["val_accuracy"]))
            except Exception:
                pass
        if "val_f1_avg" in metrics:
            try:
                TRAIN_F1.labels(framework=self.framework, split="val").set(float(metrics["val_f1_avg"]))
            except Exception:
                pass
        if "val_precision_avg" in metrics:
            try:
                TRAIN_PRECISION.labels(framework=self.framework, split="val").set(
                    float(metrics["val_precision_avg"])
                )
            except Exception:
                pass
        if "val_recall_avg" in metrics:
            try:
                TRAIN_RECALL.labels(framework=self.framework, split="val").set(float(metrics["val_recall_avg"]))
            except Exception:
                pass


class PrometheusTuneCallback(_TuneCallback):
    """Tracks Tune trial status and best metric in real-time."""

    def __init__(self, *, framework: str, metric_name: str, mode: str = "min"):
        self.framework = framework
        self.metric_name = metric_name
        self.mode = mode
        self._best: Optional[float] = None
        self._counts = {"running": 0, "succeeded": 0, "failed": 0}

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "metric_name": self.metric_name,
            "mode": self.mode,
            "_best": self._best,
            "_counts": dict(self._counts),
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.framework = state.get("framework", "unknown")
        self.metric_name = state.get("metric_name", "metric")
        self.mode = state.get("mode", "min")
        self._best = state.get("_best", None)
        self._counts = dict(state.get("_counts", {"running": 0, "succeeded": 0, "failed": 0}))

    def _publish_counts(self) -> None:
        TUNE_TRIALS_BY_STATUS.labels(framework=self.framework, status="running").set(self._counts["running"])
        TUNE_TRIALS_BY_STATUS.labels(framework=self.framework, status="succeeded").set(
            self._counts["succeeded"]
        )
        TUNE_TRIALS_BY_STATUS.labels(framework=self.framework, status="failed").set(self._counts["failed"])

    def on_trial_start(self, iteration: int, trials: Any, trial: Any, **info: Any) -> None:
        TUNE_TRIALS.labels(framework=self.framework).inc(1)
        try:
            TUNE_TRIAL_STATUS.labels(framework=self.framework, trial_id=str(trial.trial_id)).set(1)
        except Exception:
            pass
        self._counts["running"] += 1
        self._publish_counts()

    def on_trial_result(self, iteration: int, trials: Any, trial: Any, result: Any, **info: Any) -> None:
        if not isinstance(result, dict):
            return
        if self.metric_name not in result:
            return

        try:
            val = float(result[self.metric_name])
        except Exception:
            return

        improved = False
        if self._best is None:
            improved = True
        elif self.mode == "min" and val < self._best:
            improved = True
        elif self.mode == "max" and val > self._best:
            improved = True

        if improved:
            self._best = val
            TUNE_BEST_METRIC.labels(framework=self.framework, metric=self.metric_name).set(val)

    def on_trial_complete(self, iteration: int, trials: Any, trial: Any, **info: Any) -> None:
        try:
            TUNE_TRIAL_STATUS.labels(framework=self.framework, trial_id=str(trial.trial_id)).set(2)
        except Exception:
            pass
        self._counts["running"] = max(0, self._counts["running"] - 1)
        self._counts["succeeded"] += 1
        self._publish_counts()

    def on_trial_error(self, iteration: int, trials: Any, trial: Any, **info: Any) -> None:
        try:
            TUNE_TRIAL_STATUS.labels(framework=self.framework, trial_id=str(trial.trial_id)).set(3)
        except Exception:
            pass
        self._counts["running"] = max(0, self._counts["running"] - 1)
        self._counts["failed"] += 1
        self._publish_counts()
