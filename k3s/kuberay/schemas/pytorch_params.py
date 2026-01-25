"""PyTorch parameters schema.

Plain dict with common parameters for a basic supervised classifier.
Extend as needed.
"""

from __future__ import annotations
from ray import tune
from typing import Any, Dict

PYTORCH_PARAMS: Dict[str, Any] = {
    # batch size
    "batch_size": 256,
    # Training
    "max_epochs": 50,
    "lr": 1e-3,
    "weight_decay": 0.0,
}

PYTORCH_TUNE_SETTINGS: Dict[str, int] = {
    "grace_period": 5,
    "reduction_factor": 2,
    "max_epochs": 50,
}

SEARCH_SPACE_PYTORCH_PARAMS: Dict[str, Any] = {
    # Hyperparameters to tune
    "batch_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-2)}
