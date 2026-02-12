"""XGBoost tuner — concrete implementation of BaseTuner."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import ray
import ray.data
import ray.train
from ray.train.xgboost import XGBoostTrainer

from schemas.model.xgboost_params import SEARCH_SPACE_XGBOOST_PARAMS, XGBOOST_TUNE_SETTINGS
from pipeline.utils.xgboost_utils import (
    train_func,
    get_train_val_dmatrix,
    run_xgboost_train,
    RayTrainPeriodicReportCheckpointCallback,
)
from pipeline.base_tuner import BaseTuner

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Concrete tuner
# ──────────────────────────────────────────────

class XGBoostModelTuner(BaseTuner):
    """Hyperparameter tuning for XGBoost via ASHA + Ray Tune."""

    @property
    def framework_name(self) -> str:
        return "xgboost"

    @property
    def params_key(self) -> str:
        return "xgboost_params"

    @property
    def search_space(self) -> Dict[str, Any]:
        return SEARCH_SPACE_XGBOOST_PARAMS

    @property
    def tune_settings(self) -> Dict[str, Any]:
        return XGBOOST_TUNE_SETTINGS

    @property
    def tune_metric(self) -> str:
        return "validation-mlogloss"

    @property
    def tune_mode(self) -> str:
        return "min"

    @property
    def default_num_samples(self) -> int:
        return 3

    def _get_ray_trainer_cls(self):
        return XGBoostTrainer

    def _get_train_func(self):
        return train_func

    def _get_asha_max_t_key(self) -> str:
        return "num_boost_round"

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
        return {
            "target": target,
            "feature_columns": feature_columns,
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
            "num_boost_round": int(XGBOOST_TUNE_SETTINGS["num_boost_round"]),
            "xgboost_params": trial_config["xgboost_params"],
            "is_tuning": True,
        }

    def _preprocess_datasets(
        self,
        train_ds: ray.data.Dataset,
        val_ds: ray.data.Dataset,
        *,
        target: str,
        feature_columns: Optional[List[str]],
    ) -> tuple[ray.data.Dataset, ray.data.Dataset]:
        # XGBoost handles column selection inside DMatrix construction.
        return train_ds, val_ds


# ──────────────────────────────────────────────
# Module-level convenience (backward-compatible)
# ──────────────────────────────────────────────

_TUNER = XGBoostModelTuner()


def tune_model(
    table_identifier: str,
    catalog_config: dict,
    split_ranges: dict,
    target,
    feature_columns: list | None = None,
    storage_path: str = None,
    name: str = "tune",
    input_dim: int = 14,
    num_classes: int = 6,
    sample_fraction: float | None = None,
    seed: int = 42,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    extra_callbacks: list | None = None,
):
    """Backward-compatible module-level entry point."""
    return _TUNER.tune_model(
        table_identifier=table_identifier,
        catalog_config=catalog_config,
        split_ranges=split_ranges,
        target=target,
        feature_columns=feature_columns,
        storage_path=storage_path,
        name=name,
        input_dim=input_dim,
        num_classes=num_classes,
        sample_fraction=sample_fraction,
        seed=seed,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        extra_callbacks=extra_callbacks,
    )
