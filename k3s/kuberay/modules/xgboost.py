""""
XGBoost training module using Ray Train. It only supports RAM-based training."""

import os
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import xgboost
import ray.train
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

from schemas.xgboost_params import XGBOOST_PARAMS
from helpers.metrics_utils import xgb_multiclass_metrics_on_val
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train

# Training function for each worker
def train_func(config: Dict):
    """Runs on each Ray Train worker."""
    params = config.get("xgboost_params", XGBOOST_PARAMS)
    target = config["target"]
    params["num_class"] = int(config.get("num_classes", 2))
    num_boost_round = params.get("num_boost_round", 100)
    dtrain, dval = get_train_val_dmatrix(target)
    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        callbacks=[
            RayTrainReportCallback(
                metrics=["validation-mlogloss", "validation-merror"],
                frequency=1,
            )
        ],
    )

# Main training function
def train(train_dataset, val_dataset, target, storage_path, name, num_classes: int = 6, xgboost_params=None):
    scaling_config = ray.train.ScalingConfig(
        num_workers=int(os.getenv("NUM_WORKERS", 2)),
        resources_per_worker={"CPU": int(os.getenv("CPUS_PER_WORKER", 2))})
    
    params = xgboost_params if xgboost_params is not None else XGBOOST_PARAMS
    config = {
        "target": target,
        "num_classes": int(num_classes),
        "xgboost_params": params,
    }
    
    trainer = XGBoostTrainer(
        train_loop_per_worker=train_func, #Función de entrenamiento
        train_loop_config=config, #Configuración del entrenamiento
        scaling_config=scaling_config, #Configuración de recursos
        datasets={"train": train_dataset, "val": val_dataset}, #Pasar datasets leidos
        run_config=ray.train.RunConfig(storage_path=storage_path, name=name), #Donde guardar los resultados
    )

    result = trainer.fit()

    # Métricas finales (mezcla de métricas reportadas por Ray + multiclass en val)
    final_metrics: Dict[str, float] = {}
    try:
        if getattr(result, "metrics", None):
            for k, v in result.metrics.items():
                if isinstance(v, (int, float)):
                    final_metrics[k] = float(v)
    except Exception:
        pass

    try:
        if getattr(result, "checkpoint", None):
            final_metrics.update(
                xgb_multiclass_metrics_on_val(
                    val_ds=val_dataset,
                    target=target,
                    num_classes=int(num_classes),
                    booster_checkpoint=result.checkpoint,
                )
            )
    except Exception:
        pass

    return result, final_metrics