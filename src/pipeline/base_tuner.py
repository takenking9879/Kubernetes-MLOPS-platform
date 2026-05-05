"""Abstract base class for hyperparameter tuning via Ray Tune.

Defines the driver-side lifecycle shared by all frameworks:
    build scaling → build ASHA → pre-load datasets → build _trainable → build Tuner → fit() → return best config

Subclasses implement only framework-specific concerns:
    - Search space and tune settings
    - Tune metric name and mode
    - How to build the trial-level train_loop_config
    - Which Ray Trainer class to use inside each trial
    - Optional dataset preprocessing (e.g. column selection for PyTorch)

Dataset Loading Strategy:
    Datasets are loaded ONCE before the Tuner starts and shared across all trials via
    tune.with_parameters().  tune.with_parameters stores objects via ray.put(), so
    each trial receives ObjectRefs pointing to the same blocks — no data duplication.
    Materialization is obligatory — datasets are pinned in object store before trials.
    Lazy datasets remain lazy (plan is serialised as a few KB) enabling Parquet column
    pushdown and scalability to datasets larger than object store memory.
    train_loop_config propagation is preserved.
    Placement groups are not double-reserved.
"""

from __future__ import annotations

import logging
import numbers
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import ray
import ray.train
from ray import tune
from ray.train import DataConfig, ScalingConfig
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler

from src.pipeline.utils.general_utils import maybe_sample_train_ds

logger = logging.getLogger(__name__)


class BaseTuner(ABC):
    """Template for hyperparameter tuning with ASHA + Ray Tune."""

    # ──────────────────────────────────────────────
    # Abstract properties – framework identity
    # ──────────────────────────────────────────────

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Short identifier: ``"pytorch"`` or ``"xgboost"``."""

    @property
    @abstractmethod
    def params_key(self) -> str:
        """Key in param_space / trial_config, e.g. ``"pytorch_params"``."""

    @property
    @abstractmethod
    def search_space(self) -> Dict[str, Any]:
        """Ray Tune search space dict for this framework."""

    @property
    @abstractmethod
    def tune_settings(self) -> Dict[str, Any]:
        """ASHA settings: ``max_t``, ``grace_period``, ``reduction_factor``."""

    @property
    @abstractmethod
    def tune_metric(self) -> str:
        """Metric name used by ASHA scheduler, e.g. ``"val_loss"``."""

    @property
    @abstractmethod
    def tune_mode(self) -> str:
        """``"min"`` or ``"max"``."""

    @property
    @abstractmethod
    def default_num_samples(self) -> int:
        """Number of Tune trials."""

    # ──────────────────────────────────────────────
    # Abstract methods – framework-specific logic
    # ──────────────────────────────────────────────

    @abstractmethod
    def _get_ray_trainer_cls(self):
        """Return the concrete Ray Trainer class for trial runs."""

    @abstractmethod
    def _get_train_func(self):
        """Return the worker-side ``train_func`` callable."""

    @abstractmethod
    def _build_trial_train_loop_config(
        self,
        *,
        trial_config: Dict[str, Any],
        target: str,
        feature_columns: Optional[List[str]],
        input_dim: int,
        num_classes: int,
        cpus_per_worker: int,
    ) -> Dict[str, Any]:
        """Build the ``train_loop_config`` for a single trial."""

    @abstractmethod
    def _preprocess_datasets(
        self,
        train_ds: ray.data.Dataset,
        val_ds: ray.data.Dataset,
        *,
        target: str,
        feature_columns: Optional[List[str]],
    ) -> tuple[ray.data.Dataset, ray.data.Dataset]:
        """Apply framework-specific transforms (e.g. column selection).

        Return ``(train_ds, val_ds)``—possibly unchanged.
        """

    @abstractmethod
    def _get_asha_max_t_key(self) -> str:
        """Key in ``tune_settings`` used for ``max_t``, e.g. ``"max_epochs"`` or ``"num_boost_round"``."""

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _build_search_space(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge class-default search space with user overrides from params_training.yaml.

        ``override`` uses the same JSON-serialisable format sent by the frontend
        (SearchSpaceEntry in hyperparams.ts):
          {"type": "choice",     "options": [64, 128, 256]}
          {"type": "loguniform", "min": 1e-5, "max": 0.1}
          {"type": "uniform",    "min": 0.0,  "max": 1.0}
          {"type": "randint",    "min": 1,    "max": 100}
          {"type": "fixed",      "value": 256}

        Only keys that already exist in ``base`` can be overridden; unknown keys
        are silently ignored to prevent injecting unsupported parameters.
        """
        if not override:
            return base
        from ray import tune
        result = dict(base)
        for key, entry in override.items():
            if key not in result:
                continue  # only override existing params, never add new ones
            t = entry.get("type")
            if t == "choice":
                opts = entry.get("options", [])
                if opts:
                    result[key] = tune.choice(opts)
            elif t == "loguniform":
                result[key] = tune.loguniform(float(entry["min"]), float(entry["max"]))
            elif t == "uniform":
                result[key] = tune.uniform(float(entry["min"]), float(entry["max"]))
            elif t == "randint":
                result[key] = tune.randint(int(entry["min"]), int(entry["max"]))
            elif t == "fixed":
                result[key] = tune.choice([entry["value"]])
        return result

    # ──────────────────────────────────────────────
    # Concrete lifecycle – shared across frameworks
    # ──────────────────────────────────────────────

    def tune_model(
        self,
        *,
        table_identifier: str,
        catalog_config: dict,
        split_ranges: dict,
        target: str,
        feature_columns: Optional[List[str]] = None,
        storage_path: Optional[str] = None,
        name: str = "tune",
        input_dim: int = 14,
        num_classes: int = 6,
        sample_fraction: Optional[float] = None,
        seed: int = 42,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        extra_callbacks: Optional[List[object]] = None,
        number_of_trials: Optional[int] = None,
        tune_settings_override: Optional[Dict[str, Any]] = None,
        search_space_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter search. Returns ``best.config``."""

        num_workers = int(os.getenv("NUM_WORKERS_TUNE", os.getenv("NUM_WORKERS", 2)))
        cpus_per_worker = int(os.getenv("CPUS_PER_WORKER_TUNE", os.getenv("CPUS_PER_WORKER", 1)))
        gpus_per_worker = int(os.getenv("GPUS_PER_WORKER_TUNE", os.getenv("GPUS_PER_WORKER", 1)))

        # GPU detection: same pattern as BaseTrainer._resolve_gpu()
        use_gpu_env = os.getenv("USE_GPU", "auto").lower()
        if use_gpu_env == "auto":
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                gpu_available = False
        elif use_gpu_env in ("1", "true", "yes"):
            gpu_available = True
        else:
            gpu_available = False

        resources_per_worker: Dict[str, Any] = {"CPU": cpus_per_worker}
        if gpu_available:
            resources_per_worker["GPU"] = max(gpus_per_worker, 1)

        scaling_config = ScalingConfig(
            num_workers=num_workers,
            resources_per_worker=resources_per_worker,
            use_gpu=gpu_available,
        )

        # --- Hyperparameter search space ---
        # Merges class-level defaults with any per-run overrides from
        # params_training.yaml → hyperparams.search_space (sent from the UI).
        param_space = {self.params_key: self._build_search_space(self.search_space, search_space_override)}

        # --- ASHA scheduler ---
        # tune_settings_override comes from params_training.yaml → hyperparams.tuning
        # (written by the backend from RunRequest.tune_settings).  Merging here
        # makes max_epochs/num_boost_round/grace_period user-configurable per run.
        effective_tune_settings = {**self.tune_settings, **(tune_settings_override or {})}
        asha = ASHAScheduler(
            metric=self.tune_metric,
            mode=self.tune_mode,
            max_t=effective_tune_settings[self._get_asha_max_t_key()],
            grace_period=effective_tune_settings["grace_period"],
            reduction_factor=effective_tune_settings["reduction_factor"],
        )

        enable_rcs = os.getenv("ENABLE_RESOURCE_CHANGING_SCHEDULER", "false").lower() in ("1", "true", "yes")
        scheduler = ResourceChangingScheduler(base_scheduler=asha) if enable_rcs else asha

        # --- Pre-load datasets ONCE and share across all trials ---
        # Loading inside _trainable would trigger N×2 Iceberg catalog scans (N = num trials).
        # Instead, build the dataset plan once here and pass it to each trial via
        # tune.with_parameters(), which stores the datasets via ray.put() so all trials
        # reference the same object-store entries without duplicating data.
        # Lazy datasets remain lazy: only the plan (~KB) is serialised, preserving
        # Parquet column pushdown and scalability to large datasets.
        from pyiceberg.expressions import GreaterThanOrEqual, LessThanOrEqual, And
        from src.utils.baseclass import BaseUtils

        _train_range = split_ranges.get("train", {})
        _val_range = split_ranges.get("val", {})
        _t_s = BaseUtils.format_iceberg_ts(_train_range.get("start"))
        _t_e = BaseUtils.format_iceberg_ts(_train_range.get("end"))
        _v_s = BaseUtils.format_iceberg_ts(_val_range.get("start"))
        _v_e = BaseUtils.format_iceberg_ts(_val_range.get("end"))

        _train_filter = (
            And(GreaterThanOrEqual("timestamp", _t_s), LessThanOrEqual("timestamp", _t_e))
            if _t_s and _t_e else None
        )
        _val_filter = (
            And(GreaterThanOrEqual("timestamp", _v_s), LessThanOrEqual("timestamp", _v_e))
            if _v_s and _v_e else None
        )

        shared_train_ds = maybe_sample_train_ds(
            ray.data.read_iceberg(
                table_identifier=table_identifier,
                catalog_kwargs=catalog_config,
                row_filter=_train_filter,
            ),
            sample_fraction=sample_fraction,
            seed=seed,
        )
        shared_val_ds = ray.data.read_iceberg(
            table_identifier=table_identifier,
            catalog_kwargs=catalog_config,
            row_filter=_val_filter,
        )

        # Apply framework-specific column selection (lazy — pushed to source readers)
        shared_train_ds, shared_val_ds = self._preprocess_datasets(
            shared_train_ds, shared_val_ds, target=target, feature_columns=feature_columns,
        )

        # Apply row limits (lazy)
        _max_train_rows = int(os.getenv("TUNE_MAX_TRAIN_ROWS", "0"))
        _max_val_rows = int(os.getenv("TUNE_MAX_VAL_ROWS", "0"))
        if _max_train_rows > 0:
            shared_train_ds = shared_train_ds.limit(_max_train_rows)
        if _max_val_rows > 0:
            shared_val_ds = shared_val_ds.limit(_max_val_rows)

        # Materialize datasets — blocks are pinned in the object store for the full tuning run.
        # Tuning data is small (sample_fraction + row limits), so materialization is safe
        # and recommended.  Controlled by the same RAY_MATERIALIZE_DATASETS flag as training.
        if os.getenv("RAY_MATERIALIZE_DATASETS", "0").lower() in ("1", "true", "yes"):
            shared_train_ds = shared_train_ds.materialize()
            shared_val_ds = shared_val_ds.materialize()

        # --- Capture references for the closure ---
        tuner_self = self

        def _trainable(trial_config: Dict, train_ds=None, val_ds=None):
            """Tune trial function. Receives pre-loaded datasets via tune.with_parameters."""
            # Datasets are provided by the caller via tune.with_parameters.
            # Fallback to per-trial loading only when called standalone (e.g. tests).
            if train_ds is None or val_ds is None:
                raise RuntimeError(
                    "_trainable must be called via tune.with_parameters with train_ds and val_ds. "
                    "Direct invocation without datasets is not supported."
                )

            # Build train_loop_config (framework-specific)
            train_loop_config = tuner_self._build_trial_train_loop_config(
                trial_config=trial_config,
                target=target,
                feature_columns=feature_columns,
                input_dim=input_dim,
                num_classes=num_classes,
                cpus_per_worker=cpus_per_worker,
            )

            # Construct Ray Trainer for this trial
            try:
                trial_id = tune.get_context().get_trial_id()
            except Exception:
                trial_id = str(os.getpid())

            trainer_cls = tuner_self._get_ray_trainer_cls()
            trainer = trainer_cls(
                train_loop_per_worker=tuner_self._get_train_func(),
                train_loop_config=train_loop_config,
                scaling_config=scaling_config,
                datasets={"train": train_ds, "val": val_ds},
                # Only split train across workers; each worker sees the full val set
                # for correct per-trial validation metrics with NUM_WORKERS > 1.
                dataset_config=DataConfig(datasets_to_split=["train"]),
                run_config=ray.train.RunConfig(
                    storage_path=storage_path,
                    name=f"{name}_train_{trial_id}",
                ),
            )

            result = trainer.fit()

            # Report metrics back to Tune.
            # Only report essential keys — the MLflowLoggerCallback logs one HTTP call
            # per metric to the remote server. Reporting all ~50 per-class fields adds
            # ~23s of blocking I/O per trial on high-latency connections (e.g. ngrok).
            # Per-class breakdown is logged in full during final training instead.
            _TUNE_REPORT_KEYS = frozenset({
                "epoch", "epoch_time_sec",
                "train_loss", "val_loss",
                "val_accuracy",
                "val_f1_macro", "val_f1_weighted",
                "val_precision_macro", "val_recall_macro",
                "val_precision_weighted", "val_recall_weighted",
                "multiclass_metrics_time_sec",
            })
            metrics = getattr(result, "metrics", None) or {}
            report_dict: Dict[str, numbers.Real] = {}
            for k, v in metrics.items():
                if k not in _TUNE_REPORT_KEYS:
                    continue
                if not isinstance(v, numbers.Real) or isinstance(v, bool):
                    continue
                if k in ("training_iteration", "epoch", "step"):
                    report_dict[k] = int(v)
                else:
                    report_dict[k] = float(v)
            tune.report(report_dict)

        # --- MLflow callback ---
        callbacks = list(extra_callbacks or [])
        if mlflow_tracking_uri and mlflow_experiment_name:
            from ray.air.integrations.mlflow import MLflowLoggerCallback
            callbacks.append(
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=mlflow_experiment_name,
                    save_artifact=False,
                    log_params_on_trial_end=True,
                )
            )

        # --- Build and run Tuner ---
        # tune.with_parameters stores shared_train_ds and shared_val_ds via ray.put(),
        # so all trials reference the same object-store entries — no data duplication.
        tuner = tune.Tuner(
            tune.with_parameters(_trainable, train_ds=shared_train_ds, val_ds=shared_val_ds),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=number_of_trials if number_of_trials is not None else self.default_num_samples,
                scheduler=scheduler,
                max_concurrent_trials=int(os.getenv("MAX_CONCURRENT_TRIALS", "1")),
            ),
            run_config=RunConfig(
                storage_path=storage_path,
                name=name,
                callbacks=callbacks,
            ),
        )

        results = tuner.fit()
        best = results.get_best_result(metric=self.tune_metric, mode=self.tune_mode)
        return best.config
