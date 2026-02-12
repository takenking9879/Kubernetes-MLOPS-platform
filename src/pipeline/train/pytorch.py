"""PyTorch trainer — concrete implementation of BaseTrainer."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import ray.data
import ray.train
import torch
from ray.train.torch import TorchTrainer

from schemas.model.pytorch_params import PYTORCH_PARAMS
from pipeline.utils.pytorch_utils import train_func
from pipeline.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataset helpers (keep metadata columns out of torch)
# ──────────────────────────────────────────────

def _select_model_columns(
    ds: ray.data.Dataset, *, target: str, feature_columns: Optional[List[str]],
) -> ray.data.Dataset:
    """Return a Dataset with only (features + target) columns.

    Prevents ``iter_torch_batches`` from touching metadata columns
    like ``timestamp`` (numpy.datetime64), which PyTorch can't convert.
    """
    if feature_columns is None:
        return ds
    cols = list(feature_columns) + [target]
    return ds.select_columns(cols)


# ──────────────────────────────────────────────
# Driver-side evaluation
# ──────────────────────────────────────────────

def _evaluate_on_dataset(
    *,
    checkpoint: ray.train.Checkpoint,
    ds: ray.data.Dataset,
    target: str,
    feature_columns: Optional[List[str]],
    num_classes: int,
    input_dim: int,
    batch_size: int,
    prefix: str = "test",
) -> Dict[str, Any]:
    """Compute multiclass metrics from the final checkpoint on a full split."""
    import numpy as np
    from torch import nn
    from sklearn.metrics import classification_report
    from models.pytorch import NeuralNetwork

    model = NeuralNetwork(input_dim=input_dim, num_classes=num_classes)
    model.eval()

    with checkpoint.as_directory() as ckpt_dir:
        state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

    loss_fn = nn.CrossEntropyLoss()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss, total_batches = 0.0, 0

    ds_eval = _select_model_columns(ds, target=target, feature_columns=feature_columns)
    feature_cols = feature_columns

    for batch in ds_eval.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32):
        y = batch.pop(target).long()
        if feature_cols is None:
            feature_cols = sorted(batch.keys())
        X = torch.stack([batch[c] for c in feature_cols], dim=1)
        with torch.no_grad():
            logits = model(X)
            total_loss += float(loss_fn(logits, y).item())
            total_batches += 1
            y_pred = logits.argmax(dim=1)
            idx = y * num_classes + y_pred
            counts = torch.bincount(idx, minlength=num_classes * num_classes)
            conf += counts.reshape(num_classes, num_classes)

    # Metrics from confusion matrix
    from pipeline.utils.pytorch_utils import _metrics_from_confusion

    metrics: Dict[str, Any] = {}
    metrics[f"{prefix}_loss"] = total_loss / max(total_batches, 1)
    metrics.update(_metrics_from_confusion(conf, prefix=prefix))

    conf_np = conf.detach().cpu().numpy().astype(np.int64)
    metrics[f"{prefix}_confusion_matrix"] = conf_np.tolist()

    # Classification report
    total = int(conf_np.sum())
    if total > 0:
        max_rows = int(os.getenv("MLFLOW_CLASSIFICATION_REPORT_MAX_ROWS", "200000"))
        seed = int(os.getenv("SEED", "42"))
        flat = conf_np.ravel()
        if max_rows > 0 and total > max_rows:
            rng = np.random.default_rng(seed)
            p = flat / max(float(total), 1.0)
            sampled = rng.multinomial(max_rows, p)
            idx_arr = np.repeat(np.arange(flat.size, dtype=np.int64), sampled)
        else:
            idx_arr = np.repeat(np.arange(flat.size, dtype=np.int64), flat)
        y_true = (idx_arr // num_classes).astype(np.int64)
        y_pred_arr = (idx_arr % num_classes).astype(np.int64)
        metrics[f"{prefix}_classification_report"] = classification_report(
            y_true, y_pred_arr, labels=list(range(num_classes)), digits=4, zero_division=0,
        )

    return metrics


# ──────────────────────────────────────────────
# Concrete trainer
# ──────────────────────────────────────────────

class PyTorchModelTrainer(BaseTrainer):
    """PyTorch distributed training via Ray TorchTrainer."""

    @property
    def framework_name(self) -> str:
        return "pytorch"

    @property
    def params_key(self) -> str:
        return "pytorch_params"

    @property
    def default_params(self) -> Dict[str, Any]:
        return PYTORCH_PARAMS

    def _get_ray_trainer_cls(self):
        return TorchTrainer

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
    ) -> Dict[str, Any]:
        return {
            "target": target,
            "feature_columns": feature_columns,
            "pytorch_params": params,
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
            "is_tuning": False,
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
        train_ds = _select_model_columns(train_ds, target=target, feature_columns=feature_columns)
        val_ds = _select_model_columns(val_ds, target=target, feature_columns=feature_columns)
        if test_ds is not None:
            test_ds = _select_model_columns(test_ds, target=target, feature_columns=feature_columns)
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
    ) -> Dict[str, Any]:
        return _evaluate_on_dataset(
            checkpoint=result.checkpoint,
            ds=ds,
            target=target,
            feature_columns=feature_columns,
            num_classes=int(num_classes),
            input_dim=int(input_dim),
            batch_size=int(params.get("batch_size", 256)),
            prefix=prefix,
        )

    def _build_scaling_config(self) -> ray.train.ScalingConfig:
        """PyTorch adds GPU detection to the base scaling config."""
        return ray.train.ScalingConfig(
            num_workers=int(os.getenv("NUM_WORKERS", 2)),
            resources_per_worker={"CPU": int(os.getenv("CPUS_PER_WORKER", 2))},
            use_gpu=torch.cuda.is_available(),
        )


# ──────────────────────────────────────────────
# Module-level convenience (backward-compatible)
# ──────────────────────────────────────────────

_TRAINER = PyTorchModelTrainer()


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
    pytorch_params=None,
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
        params=pytorch_params,
        callbacks=callbacks,
    )
