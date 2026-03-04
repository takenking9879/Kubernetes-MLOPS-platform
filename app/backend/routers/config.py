"""GET /api/v2/config — static training capabilities manifest.

Returns the available frameworks, task types (with loss function and
metric descriptions), and model architectures.  The data mirrors the
Python registries (pipeline/tasks.py, models/registry.py) so the
frontend never hardcodes these choices.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v2/config", tags=["config"])

_CONFIG: dict = {
    "frameworks": [
        {
            "id": "xgboost",
            "label": "XGBoost",
            "description": "Gradient-boosted trees. Fast, tabular, CPU-efficient.",
        },
        {
            "id": "pytorch",
            "label": "PyTorch (MLP)",
            "description": "Multi-layer perceptron via Ray TorchTrainer.",
        },
    ],
    "task_types": [
        {
            "id": "classification",
            "label": "Classification",
            "description": "Predict a discrete class label.",
            "loss": {
                "pytorch": {
                    "name": "CrossEntropyLoss",
                    "description": "Log-loss over class probabilities. Lower is better. No upper bound.",
                },
                "xgboost": {
                    "name": "mlogloss (multi:softprob)",
                    "description": "Multi-class log-loss. Lower is better. 0 = perfect (unachievable in practice).",
                },
            },
            "metrics": [
                {
                    "key": "accuracy",
                    "label": "Accuracy",
                    "range": "[0, 1]",
                    "perfect": 1,
                    "note": "Fraction of predictions that were correct. Misleading on class-imbalanced data.",
                },
                {
                    "key": "f1_macro",
                    "label": "F1 Macro",
                    "range": "[0, 1]",
                    "perfect": 1,
                    "note": "Harmonic mean of precision and recall, averaged equally across all classes. Best metric for imbalanced datasets.",
                },
                {
                    "key": "f1_weighted",
                    "label": "F1 Weighted",
                    "range": "[0, 1]",
                    "perfect": 1,
                    "note": "F1 averaged across classes weighted by support (number of true instances). Penalises errors on larger classes.",
                },
                {
                    "key": "precision_macro",
                    "label": "Precision Macro",
                    "range": "[0, 1]",
                    "perfect": 1,
                    "note": "Of all positive predictions, the fraction that were actually positive. High precision = few false alarms.",
                },
                {
                    "key": "recall_macro",
                    "label": "Recall Macro",
                    "range": "[0, 1]",
                    "perfect": 1,
                    "note": "Of all actual positives, the fraction the model caught. High recall = few missed detections.",
                },
            ],
        },
        {
            "id": "regression",
            "label": "Regression",
            "description": "Predict a continuous numeric value.",
            "loss": {
                "pytorch": {
                    "name": "MSELoss",
                    "description": "Mean squared error. Penalises large errors heavily — sensitive to outliers. Lower is better.",
                },
                "xgboost": {
                    "name": "RMSE (reg:squarederror)",
                    "description": "Root mean squared error. In the same units as the target. Lower is better.",
                },
            },
            "metrics": [
                {
                    "key": "mse",
                    "label": "MSE",
                    "range": "[0, ∞)",
                    "perfect": 0,
                    "note": "Mean squared error. Penalises large errors quadratically — very sensitive to outliers.",
                },
                {
                    "key": "rmse",
                    "label": "RMSE",
                    "range": "[0, ∞)",
                    "perfect": 0,
                    "note": "Square root of MSE. Same units as the target variable — easier to interpret than MSE.",
                },
                {
                    "key": "mae",
                    "label": "MAE",
                    "range": "[0, ∞)",
                    "perfect": 0,
                    "note": "Mean absolute error. Robust to outliers. Reports average deviation in target units.",
                },
                {
                    "key": "r2",
                    "label": "R²",
                    "range": "(-∞, 1]",
                    "perfect": 1,
                    "note": "Fraction of variance explained. 1 = perfect fit, 0 = predicts the mean, <0 = worse than predicting the mean.",
                },
            ],
        },
    ],
    "model_types": [
        {
            "id": "mlp",
            "label": "MLP",
            "framework": "pytorch",
            "description": "Multi-layer perceptron. Configurable hidden layers. General-purpose tabular baseline.",
        },
    ],
}


@router.get("")
async def get_training_config() -> dict:
    """Return the static manifest of available frameworks, task types, and model types."""
    return _CONFIG
