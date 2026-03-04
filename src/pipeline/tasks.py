"""Task abstraction: classification vs regression (extensible).

Provides a registry of task configurations that decouple loss functions,
evaluation metrics, and prediction logic from the training loop.

Usage::

    from pipeline.tasks import get_task_config

    task_config = get_task_config("classification")
    loss_cls = getattr(nn, task_config.torch_loss_cls)
    loss_fn = loss_cls(**task_config.torch_loss_kwargs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a training task type."""

    task_type: TaskType

    # PyTorch loss class name (from torch.nn) and optional kwargs.
    # Instantiated in train_func via: getattr(nn, torch_loss_cls)(**torch_loss_kwargs)
    torch_loss_cls: str
    torch_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # XGBoost objective and eval metrics.
    xgb_objective: str = ""
    xgb_eval_metric: List[str] = field(default_factory=list)

    # How to derive predictions from raw model output.
    #   "argmax"  — classification: y_pred = output.argmax(dim=1)
    #   "raw"     — regression:     y_pred = output.squeeze(-1)
    prediction_mode: str = "argmax"


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "classification": TaskConfig(
        task_type="classification",
        torch_loss_cls="CrossEntropyLoss",
        xgb_objective="multi:softprob",
        xgb_eval_metric=["mlogloss", "merror"],
        prediction_mode="argmax",
    ),
    "regression": TaskConfig(
        task_type="regression",
        torch_loss_cls="MSELoss",
        xgb_objective="reg:squarederror",
        xgb_eval_metric=["rmse", "mae"],
        prediction_mode="raw",
    ),
}


def get_task_config(task_type: str) -> TaskConfig:
    """Return the ``TaskConfig`` for *task_type*.

    Raises ``ValueError`` for unknown task types.
    """
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. "
            f"Supported: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_type]
