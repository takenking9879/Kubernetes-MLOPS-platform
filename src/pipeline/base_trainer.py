"""Abstract base class for distributed model training via Ray Train.

Defines the driver-side lifecycle shared by all frameworks:
    build config → create Ray Trainer → fit() → extract metrics → evaluate splits → return

Subclasses implement only framework-specific concerns:
    - Which Ray Trainer class to use (TorchTrainer vs XGBoostTrainer)
    - How to build the train_loop_config dict
    - How to preprocess Ray Datasets before passing to the Trainer
    - How to evaluate a dataset split after training
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import ray.train

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Template for a single distributed training run (no tuning)."""

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
        """Key used in train_loop_config, e.g. ``"pytorch_params"``."""

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Fallback hyperparameters when none are supplied."""

    # ──────────────────────────────────────────────
    # Abstract methods – framework-specific logic
    # ──────────────────────────────────────────────

    @abstractmethod
    def _get_ray_trainer_cls(self):
        """Return the concrete Ray *Trainer* class (e.g. ``TorchTrainer``)."""

    @abstractmethod
    def _get_train_func(self):
        """Return the worker-side ``train_func`` callable."""

    @abstractmethod
    def _build_train_loop_config(
        self,
        *,
        target: str,
        feature_columns: Optional[List[str]],
        params: Dict[str, Any],
        input_dim: int,
        num_classes: int,
        cpus_per_worker: int,
    ) -> Dict[str, Any]:
        """Build the full ``train_loop_config`` dict (framework-specific keys included)."""

    @abstractmethod
    def _preprocess_datasets(
        self,
        train_ds: ray.data.Dataset,
        val_ds: ray.data.Dataset,
        test_ds: Optional[ray.data.Dataset],
        *,
        target: str,
        feature_columns: Optional[List[str]],
    ) -> Tuple[ray.data.Dataset, ray.data.Dataset, Optional[ray.data.Dataset]]:
        """Apply framework-specific dataset transformations (e.g. column selection for PyTorch).

        Return ``(train_ds, val_ds, test_ds)``—possibly unchanged.
        """

    @abstractmethod
    def _evaluate_split(
        self,
        *,
        result: ray.train.Result,
        ds: ray.data.Dataset,
        target: str,
        feature_columns: Optional[List[str]],
        num_classes: int,
        input_dim: int,
        params: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """Compute final metrics on a full dataset split (driver-side)."""

    # ──────────────────────────────────────────────
    # Concrete lifecycle – shared across frameworks
    # ──────────────────────────────────────────────

    def _build_scaling_config(self) -> ray.train.ScalingConfig:
        return ray.train.ScalingConfig(
            num_workers=int(os.getenv("NUM_WORKERS", 2)),
            resources_per_worker={"CPU": int(os.getenv("CPUS_PER_WORKER", 2))},
        )

    def _extract_result_metrics(self, result: ray.train.Result) -> Dict[str, Any]:
        """Pull numeric metrics from ``result.metrics``."""
        metrics: Dict[str, Any] = {}
        raw = getattr(result, "metrics", None)
        if raw:
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        return metrics

    def train(
        self,
        *,
        train_dataset: ray.data.Dataset,
        val_dataset: ray.data.Dataset,
        test_dataset: Optional[ray.data.Dataset],
        target: str,
        storage_path: str,
        name: str,
        feature_columns: Optional[List[str]] = None,
        input_dim: int = 14,
        num_classes: int = 6,
        params: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[object]] = None,
    ) -> Tuple[ray.train.Result, Dict[str, Any]]:
        """Execute the full training lifecycle.

        Returns ``(result, final_metrics)``.
        """
        effective_params = params if params is not None else dict(self.default_params)
        cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", 2))

        # 1 — Preprocess datasets (framework-specific)
        train_dataset, val_dataset, test_dataset = self._preprocess_datasets(
            train_dataset, val_dataset, test_dataset,
            target=target,
            feature_columns=feature_columns,
        )

        # 2 — Build config dict for worker train_func
        config = self._build_train_loop_config(
            target=target,
            feature_columns=feature_columns,
            params=effective_params,
            input_dim=input_dim,
            num_classes=num_classes,
            cpus_per_worker=cpus_per_worker,
        )

        # 3 — Construct Ray Trainer
        trainer_cls = self._get_ray_trainer_cls()
        trainer = trainer_cls(
            train_loop_per_worker=self._get_train_func(),
            train_loop_config=config,
            scaling_config=self._build_scaling_config(),
            datasets={"train": train_dataset, "val": val_dataset},
            run_config=ray.train.RunConfig(
                storage_path=storage_path,
                name=name,
                callbacks=callbacks or None,
            ),
        )

        # 4 — Fit
        start_time = time.perf_counter()
        result = trainer.fit()
        train_time_sec = time.perf_counter() - start_time
        print(f"[{self.framework_name}] distributed train_time_sec={train_time_sec:.2f}")

        # 5 — Collect metrics reported by Ray Train
        final_metrics = self._extract_result_metrics(result)

        # 6 — Driver-side evaluation on full splits
        mc_start = time.perf_counter()

        final_metrics.update(
            self._evaluate_split(
                result=result,
                ds=val_dataset,
                target=target,
                feature_columns=feature_columns,
                num_classes=num_classes,
                input_dim=input_dim,
                params=effective_params,
                prefix="val",
            )
        )

        if test_dataset is not None:
            final_metrics.update(
                self._evaluate_split(
                    result=result,
                    ds=test_dataset,
                    target=target,
                    feature_columns=feature_columns,
                    num_classes=num_classes,
                    input_dim=input_dim,
                    params=effective_params,
                    prefix="test",
                )
            )

        final_metrics["multiclass_metrics_time_sec"] = time.perf_counter() - mc_start
        final_metrics["train_time_sec"] = train_time_sec

        return result, final_metrics
