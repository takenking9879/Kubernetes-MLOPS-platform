"""XGBoost trainer — concrete implementation of BaseTrainer.

Uses QuantileDMatrix with streaming iterator for memory-efficient training."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import ray.data
import ray.train
from ray.train.xgboost import XGBoostTrainer

from schemas.model.xgboost_params import XGBOOST_PARAMS
from pipeline.utils.metrics_utils import xgb_multiclass_metrics_on_ds, xgb_regression_metrics_on_ds
from pipeline.utils.xgboost_utils import (
    train_func,
    get_train_val_dmatrix,
    run_xgboost_train,
    RayTrainPeriodicReportCheckpointCallback,
)
from pipeline.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Concrete trainer
# ──────────────────────────────────────────────

class XGBoostModelTrainer(BaseTrainer):
    """XGBoost distributed training via Ray XGBoostTrainer."""

    @property
    def framework_name(self) -> str:
        return "xgboost"

    @property
    def params_key(self) -> str:
        return "xgboost_params"

    @property
    def default_params(self) -> Dict[str, Any]:
        return XGBOOST_PARAMS

    def _get_ray_trainer_cls(self):
        return XGBoostTrainer

    def _get_train_func(self):
        return train_func

    def _build_train_loop_config(
        self,
        *,
        target: str,
        feature_columns: Optional[List[str]],
        params: Dict[str, Any],
        input_dim: int,
        num_classes: int,
        cpus_per_worker: int,
        task_type: str = "classification",
        model_type: str = "mlp",
    ) -> Dict[str, Any]:
        return {
            "target": target,
            "feature_columns": feature_columns,
            "xgboost_params": params,
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
            "is_tuning": False,
            "task_type": task_type,
            "model_type": model_type,
        }

    def _preprocess_datasets(
        self,
        train_ds: ray.data.Dataset,
        val_ds: ray.data.Dataset,
        test_ds: Optional[ray.data.Dataset],
        *,
        target: str,
        feature_columns: Optional[List[str]],
    ) -> Tuple[ray.data.Dataset, ray.data.Dataset, Optional[ray.data.Dataset]]:
        # XGBoost handles column selection via logic in get_train_val_dmatrix
        # inside the worker train_func, so no driver-side preprocessing needed.
        return train_ds, val_ds, test_ds

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
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        if task_type == "classification":
            return xgb_multiclass_metrics_on_ds(
                ds=ds,
                split=prefix,
                target=target,
                feature_columns=feature_columns,
                num_classes=int(num_classes),
                booster_checkpoint=result.checkpoint,
            )
        return xgb_regression_metrics_on_ds(
            ds=ds,
            split=prefix,
            target=target,
            feature_columns=feature_columns,
            booster_checkpoint=result.checkpoint,
        )


# ──────────────────────────────────────────────
# Module-level convenience (backward-compatible)
# ──────────────────────────────────────────────

_TRAINER = XGBoostModelTrainer()


def train(
    train_dataset,
    val_dataset,
    test_dataset,
    target,
    storage_path,
    name,
    feature_columns: Optional[List[str]] = None,
    input_dim: int = 14,
    num_classes: int = 6,
    xgboost_params=None,
    callbacks: Optional[List[object]] = None,
):
    """Backward-compatible module-level entry point."""
    return _TRAINER.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        target=target,
        storage_path=storage_path,
        name=name,
        feature_columns=feature_columns,
        input_dim=input_dim,
        num_classes=num_classes,
        params=xgboost_params,
        callbacks=callbacks,
    )
