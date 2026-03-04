from __future__ import annotations

import os
import tempfile
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import xgboost
from ray.train.xgboost import RayTrainReportCallback
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger_std = logging.getLogger(__name__)


# ── Regression metrics ───────────────────────────────────────────────────────


def regression_metrics_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "val",
) -> Dict[str, float]:
    """Compute regression metrics from arrays of true and predicted values."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    rmse = float(np.sqrt(mse))

    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2,
    }


def xgb_regression_metrics_on_ds(
    *,
    ds,
    split: str,
    target: str,
    feature_columns: Optional[List[str]] = None,
    booster_checkpoint,
) -> Dict[str, Any]:
    """Compute regression metrics for XGBoost on a Ray Dataset split.

    Aggregates predictions batch-by-batch and computes final metrics on the driver.
    """
    try:
        booster = RayTrainReportCallback.get_model(booster_checkpoint)
        model_bytes = booster.save_raw()

        def predict_batch(df: "pd.DataFrame") -> "pd.DataFrame":
            y_true = df[target].astype("float64").to_numpy()
            if feature_columns:
                X = df[feature_columns]
            else:
                X = df.drop(columns=[target], errors="ignore").select_dtypes(
                    include=[np.number, "bool"]
                )

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ubj")
            try:
                tmp.write(model_bytes)
                tmp.close()
                b = xgboost.Booster()
                b.load_model(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            dmat = xgboost.DMatrix(X)
            y_pred = b.predict(dmat).astype("float64")

            return pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        result_rows = ds.map_batches(predict_batch, batch_format="pandas").take_all()

        all_true = np.concatenate([np.atleast_1d(r["y_true"]) for r in result_rows])
        all_pred = np.concatenate([np.atleast_1d(r["y_pred"]) for r in result_rows])

        return regression_metrics_np(all_true, all_pred, prefix=split)

    except Exception as e:
        logger_std.error(
            "Error computing regression metrics for XGBoost: %s",
            str(e),
            exc_info=True,
        )
        return {}


# ── Classification metrics ───────────────────────────────────────────────────


def metrics_from_confusion_np(conf, *, prefix: str = "val") -> Dict[str, float]:
    """Compute classification-report-like metrics from a confusion matrix.

    Expected conf shape: [C, C] where rows=true labels, cols=pred labels.
    """
    conf = np.asarray(conf, dtype=np.int64)
    support = conf.sum(axis=1)
    tp = np.diag(conf)
    pred_sum = conf.sum(axis=0)

    precision = np.divide(tp, np.maximum(pred_sum, 1), dtype=np.float64)
    recall = np.divide(tp, np.maximum(support, 1), dtype=np.float64)
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-12), dtype=np.float64)

    accuracy = float(tp.sum() / max(conf.sum(), 1))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    weights = support / max(support.sum(), 1)
    weighted_precision = float(np.sum(precision * weights))
    weighted_recall = float(np.sum(recall * weights))
    weighted_f1 = float(np.sum(f1 * weights))

    metrics: Dict[str, float] = {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision_macro": macro_precision,
        f"{prefix}_recall_macro": macro_recall,
        f"{prefix}_f1_macro": macro_f1,
        # User aliases for easier interpretation
        f"{prefix}_precision_avg": macro_precision,
        f"{prefix}_recall_avg": macro_recall,
        f"{prefix}_f1_avg": macro_f1,
        f"{prefix}_precision_weighted": weighted_precision,
        f"{prefix}_recall_weighted": weighted_recall,
        f"{prefix}_f1_weighted": weighted_f1,
    }

    for i in range(conf.shape[0]):
        metrics[f"{prefix}_precision_class_{i}"] = float(precision[i])
        metrics[f"{prefix}_recall_class_{i}"] = float(recall[i])
        metrics[f"{prefix}_f1_class_{i}"] = float(f1[i])
        metrics[f"{prefix}_support_class_{i}"] = float(support[i])

    return metrics


def xgb_multiclass_metrics_on_ds(
    *,
    ds,
    split: str,
    target: str,
    feature_columns: Optional[List[str]] = None,
    num_classes: int,
    booster_checkpoint,
) -> Dict[str, Any]:
    """Compute multiclass metrics for XGBoost on a Ray Dataset split.

    This avoids collecting the full dataset to the driver by aggregating a confusion matrix.
    """

    try:
        # Ray Train stores XGBoost models inside a generic `ray.train.Checkpoint`.
        # Per Ray docs, use RayTrainReportCallback.get_model(checkpoint) to load it.
        booster = RayTrainReportCallback.get_model(booster_checkpoint)
        model_bytes = booster.save_raw()

        # NOTE: The previous implementation used `groupby(...).count()` which forces
        # a shuffle + hash aggregate (slow for small/medium datasets on Kubernetes).
        # Instead, compute a confusion matrix per batch and reduce on the driver.

        def predict_and_cm_batch(df: "pd.DataFrame") -> "pd.DataFrame":
            y_true = df[target].astype("int64").to_numpy()
            if feature_columns:
                X = df[feature_columns]
            else:
                # Fallback: drop label + keep only numeric/bool columns.
                # Avoid metadata columns like timestamps (datetime64) which break DMatrix.
                X = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number, "bool"])

            # Load model from bytes inside the worker
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ubj")
            try:
                tmp.write(model_bytes)
                tmp.close()
                b = xgboost.Booster()
                b.load_model(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception as e:
                    logger_std.debug(
                        "No se pudo borrar archivo temporal %s: %s",
                        tmp.name,
                        str(e),
                        exc_info=True,
                    )

            dmat = xgboost.DMatrix(X)
            probs = b.predict(dmat)
            if probs.ndim == 1:
                y_pred = (probs > 0.5).astype("int64")
            else:
                y_pred = probs.argmax(axis=1).astype("int64")

            # Compute confusion matrix counts for this batch.
            # Vectorized bincount avoids Python loops.
            mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
            yt = y_true[mask].astype(np.int64, copy=False)
            yp = y_pred[mask].astype(np.int64, copy=False)
            idx = yt * num_classes + yp
            cm = np.bincount(idx, minlength=num_classes * num_classes).reshape((num_classes, num_classes))

            # One row per batch: store flattened counts.
            return pd.DataFrame({"cm": [cm.ravel().tolist()]})

        cm_rows = ds.map_batches(predict_and_cm_batch, batch_format="pandas").take_all()
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for r in cm_rows:
            flat = np.asarray(r["cm"], dtype=np.int64)
            if flat.size != num_classes * num_classes:
                continue
            conf += flat.reshape((num_classes, num_classes))

        out: Dict[str, Any] = metrics_from_confusion_np(conf, prefix=split)
        out[f"{split}_confusion_matrix"] = conf.tolist()

        # Build y_true/y_pred for sklearn classification_report.
        # If very large, sample pairs from confusion matrix distribution.
        try:
            total = int(conf.sum())
            max_rows = int(os.getenv("MLFLOW_CLASSIFICATION_REPORT_MAX_ROWS", "200000"))
            seed = int(os.getenv("SEED", "42"))
            flat = conf.ravel()
            if total > 0:
                if max_rows > 0 and total > max_rows:
                    rng = np.random.default_rng(seed)
                    p = flat / max(float(total), 1.0)
                    sampled = rng.multinomial(max_rows, p)
                    idx = np.repeat(np.arange(flat.size, dtype=np.int64), sampled)
                else:
                    idx = np.repeat(np.arange(flat.size, dtype=np.int64), flat)

                y_true = (idx // num_classes).astype(np.int64)
                y_pred = (idx % num_classes).astype(np.int64)
                out[f"{split}_classification_report"] = classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    digits=4,
                    zero_division=0,
                )
        except Exception as e:
            logger_std.warning(
                "No se pudo generar classification_report para XGBoost: %s",
                str(e),
                exc_info=True,
            )

        return out

    except Exception as e:
        logger_std.error(
            f"Error calculando métricas multiclass de XGBoost: {str(e)}",
            exc_info=True,
        )
        return {}


def xgb_multiclass_metrics_on_val(
    *,
    val_ds,
    target: str,
    num_classes: int,
    booster_checkpoint,
) -> Dict[str, Any]:
    """Backward-compatible wrapper (validation split)."""

    return xgb_multiclass_metrics_on_ds(
        ds=val_ds,
        split="val",
        target=target,
        num_classes=num_classes,
        booster_checkpoint=booster_checkpoint,
    )
