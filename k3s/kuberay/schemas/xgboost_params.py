"""XGBoost parameters used by Ray Train/Tune.

This repo uses:
- `XGBOOST_PARAMS` as the default training parameters.
- `SEARCH_SPACE_XGBOOST_PARAMS` as the Ray Tune search space for hyperparameter
  tuning.
"""

from __future__ import annotations
from typing import Any, Dict
from ray import tune

# Default (non-tuned) params used for final distributed training.
XGBOOST_PARAMS: Dict[str, Any] = {
    "num_boost_round": 10,
    "objective": "multi:softprob",
    "eval_metric": ["mlogloss", "merror"],
    "booster": "gbtree",
    "tree_method": "hist",
    "verbosity": 1,
    "eta": 0.3,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
}

# Ray Tune scheduler settings.
XGBOOST_TUNE_SETTINGS: Dict[str, int] = {
  "num_boost_round": 50,
  "grace_period": 5,
  "reduction_factor": 2,
}


# Ray Tune search space for cheap HPT (ONLY XGBoost params).
SEARCH_SPACE_XGBOOST_PARAMS: Dict[str, Any] = {
    "objective": "multi:softprob",
    "eval_metric": ["mlogloss", "merror"],
    "max_depth": tune.randint(3, 11),
    "min_child_weight": tune.choice([1, 2, 3]),
    "subsample": tune.uniform(0.5, 1.0),
    "eta": tune.loguniform(1e-4, 3e-1),
    "lambda": tune.loguniform(1e-3, 10.0),
    "alpha": tune.loguniform(1e-3, 10.0),
}