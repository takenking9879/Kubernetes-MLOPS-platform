"""Abstract base class for hyperparameter tuning via Ray Tune.

Defines the driver-side lifecycle shared by all frameworks:
    build scaling → build ASHA → build _trainable → build Tuner → fit() → return best config

Subclasses implement only framework-specific concerns:
    - Search space and tune settings
    - Tune metric name and mode
    - How to build the trial-level train_loop_config
    - Which Ray Trainer class to use inside each trial
    - Optional dataset preprocessing (e.g. column selection for PyTorch)

Ray Constraints Respected:
    - Datasets are loaded inside _trainable (not passed via tune.with_parameters)
    - train_loop_config propagation is preserved
    - Placement groups are not double-reserved
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
from ray.train import ScalingConfig
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

        # --- Capture references for the closure ---
        tuner_self = self

        def _trainable(trial_config: Dict):
            """Tune trial function. Loads data from Iceberg inside the trial."""
            from pyiceberg.expressions import GreaterThanOrEqual, LessThanOrEqual, And
            from src.utils.baseclass import BaseUtils

            # Build Iceberg row filters from split ranges
            train_range = split_ranges.get("train", {})
            val_range = split_ranges.get("val", {})

            t_s = BaseUtils.format_iceberg_ts(train_range.get("start"))
            t_e = BaseUtils.format_iceberg_ts(train_range.get("end"))
            v_s = BaseUtils.format_iceberg_ts(val_range.get("start"))
            v_e = BaseUtils.format_iceberg_ts(val_range.get("end"))

            train_filter = (
                And(GreaterThanOrEqual("timestamp", t_s), LessThanOrEqual("timestamp", t_e))
                if t_s and t_e else None
            )
            val_filter = (
                And(GreaterThanOrEqual("timestamp", v_s), LessThanOrEqual("timestamp", v_e))
                if v_s and v_e else None
            )

            # Load datasets from Iceberg (cannot be passed via tune.with_parameters)
            train_ds = maybe_sample_train_ds(
                ray.data.read_iceberg(
                    table_identifier=table_identifier,
                    catalog_kwargs=catalog_config,
                    row_filter=train_filter,
                ),
                sample_fraction=sample_fraction,
                seed=seed,
            )
            val_ds = ray.data.read_iceberg(
                table_identifier=table_identifier,
                catalog_kwargs=catalog_config,
                row_filter=val_filter,
            )

            # Framework-specific dataset preprocessing
            train_ds, val_ds = tuner_self._preprocess_datasets(
                train_ds, val_ds, target=target, feature_columns=feature_columns,
            )

            # Optional row limits
            max_train_rows = int(os.getenv("TUNE_MAX_TRAIN_ROWS", "0"))
            max_val_rows = int(os.getenv("TUNE_MAX_VAL_ROWS", "0"))
            if max_train_rows > 0:
                train_ds = train_ds.limit(max_train_rows)
            if max_val_rows > 0:
                val_ds = val_ds.limit(max_val_rows)

            # Optional materialization
            if os.getenv("RAY_MATERIALIZE_DATASETS_TUNE", "0").lower() in ("1", "true", "yes"):
                train_ds = train_ds.materialize()
                val_ds = val_ds.materialize()

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
                run_config=ray.train.RunConfig(
                    storage_path=storage_path,
                    name=f"{name}_train_{trial_id}",
                ),
            )

            result = trainer.fit()

            # Report metrics back to Tune
            metrics = getattr(result, "metrics", None) or {}
            report_dict: Dict[str, numbers.Real] = {}
            for k, v in metrics.items():
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
        tuner = tune.Tuner(
            _trainable,
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
