"""Prometheus callback for Ray Tune hyperparameter tuning."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .metrics import (
    TUNE_TRIALS,
    TUNE_TRIAL_STATUS,
    TUNE_BEST_METRIC,
    TUNE_TRIALS_BY_STATUS,
)

try:
    from ray.tune import Callback as _TuneCallback
except Exception:  # pragma: no cover
    _TuneCallback = object  # type: ignore


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
