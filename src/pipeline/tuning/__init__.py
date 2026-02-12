"""Framework registry for tuning modules.

Usage from main.py::

    from src.pipeline.tuning import get_tuner

    tuner = get_tuner("pytorch")             # → PyTorchModelTuner instance
    best_config = tuner.tune_model(...)

The module-level ``tune_model()`` functions remain for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.base_tuner import BaseTuner


def get_tuner(framework: str) -> "BaseTuner":
    """Return a ``BaseTuner`` subclass instance for the given framework.

    Raises ``ValueError`` for unsupported frameworks.
    """
    if framework == "pytorch":
        from pipeline.tuning.pytorch import PyTorchModelTuner
        return PyTorchModelTuner()

    if framework == "xgboost":
        from pipeline.tuning.xgboost import XGBoostModelTuner
        return XGBoostModelTuner()

    raise ValueError(
        f"Unsupported tuning framework: {framework!r}. "
        f"Supported: 'pytorch', 'xgboost'."
    )
