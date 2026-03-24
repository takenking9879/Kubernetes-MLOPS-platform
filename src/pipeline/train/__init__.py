"""Framework registry for training modules.

Usage from main.py::

    from src.pipeline.train import get_trainer

    trainer = get_trainer("pytorch")         # → PyTorchModelTrainer instance
    result, metrics = trainer.train(...)

The module-level ``train()`` functions remain for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.base_trainer import BaseTrainer


def get_trainer(framework: str) -> "BaseTrainer":
    """Return a ``BaseTrainer`` subclass instance for the given framework.

    Raises ``ValueError`` for unsupported frameworks.
    """
    if framework in ("pytorch", "ssm", "bae"):
        from pipeline.train.pytorch import PyTorchModelTrainer
        return PyTorchModelTrainer()

    if framework == "xgboost":
        from pipeline.train.xgboost import XGBoostModelTrainer
        return XGBoostModelTrainer()

    raise ValueError(
        f"Unsupported training framework: {framework!r}. "
        f"Supported: 'pytorch', 'xgboost', 'ssm', 'bae'."
    )
