"""PyTorch tuner — concrete implementation of BaseTuner."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import ray
import ray.data
from ray.train.torch import TorchTrainer

from schemas.model.pytorch_params import SEARCH_SPACE_PYTORCH_PARAMS, PYTORCH_TUNE_SETTINGS
from pipeline.utils.pytorch_utils import train_func
from pipeline.base_tuner import BaseTuner


# ──────────────────────────────────────────────
# Dataset helper (same as train module)
# ──────────────────────────────────────────────

def _select_model_columns(
    ds: ray.data.Dataset, *, target: str, feature_columns: Optional[List[str]],
) -> ray.data.Dataset:
    if feature_columns is None:
        return ds
    cols = list(feature_columns) + [target]
    return ds.select_columns(cols)


# ──────────────────────────────────────────────
# Concrete tuner
# ──────────────────────────────────────────────

class PyTorchModelTuner(BaseTuner):
    """Hyperparameter tuning for PyTorch via ASHA + Ray Tune."""

    @property
    def framework_name(self) -> str:
        return "pytorch"

    @property
    def params_key(self) -> str:
        return "pytorch_params"

    @property
    def search_space(self) -> Dict[str, Any]:
        return SEARCH_SPACE_PYTORCH_PARAMS

    @property
    def tune_settings(self) -> Dict[str, Any]:
        return PYTORCH_TUNE_SETTINGS

    @property
    def tune_metric(self) -> str:
        return "val_loss"

    @property
    def tune_mode(self) -> str:
        return "min"

    @property
    def default_num_samples(self) -> int:
        return 5

    def _get_ray_trainer_cls(self):
        return TorchTrainer

    def _get_train_func(self):
        return train_func

    def _get_asha_max_t_key(self) -> str:
        return "max_epochs"

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
            "pytorch_params": trial_config["pytorch_params"],
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
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
        train_ds = _select_model_columns(train_ds, target=target, feature_columns=feature_columns)
        val_ds = _select_model_columns(val_ds, target=target, feature_columns=feature_columns)
        return train_ds, val_ds


# ──────────────────────────────────────────────
# Module-level convenience (backward-compatible)
# ──────────────────────────────────────────────

_TUNER = PyTorchModelTuner()


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
