from typing import Tuple

import ray
import xgboost


def get_train_val_dmatrix(target: str) -> Tuple[xgboost.DMatrix, xgboost.DMatrix]:
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    train_df = train_shard.materialize().to_pandas()
    val_df = val_shard.materialize().to_pandas()

    train_X = train_df.drop(columns=target)
    train_y = train_df[target]
    val_X = val_df.drop(columns=target)
    val_y = val_df[target]

    return xgboost.DMatrix(train_X, label=train_y), xgboost.DMatrix(val_X, label=val_y)


def run_xgboost_train(
    *,
    params: dict,
    dtrain: xgboost.DMatrix,
    dval: xgboost.DMatrix,
    callbacks: list,
    num_boost_round: int,
    xgb_model=None,
):
    return xgboost.train(
        params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        verbose_eval=False,
        xgb_model=xgb_model,
        callbacks=callbacks,
    )
