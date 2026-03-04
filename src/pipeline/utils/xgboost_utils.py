from typing import Tuple, Dict, Optional, List
import os
import time
import logging
import tempfile
import ray
import xgboost
from ray.train import Checkpoint

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from schemas.model.xgboost_params import XGBOOST_PARAMS
from pipeline.tasks import get_task_config

logger = logging.getLogger(__name__)

# Prometheus (optional; enabled by default for training observability)
try:  # pragma: no cover
    from prometheus_client import start_http_server
    from src.prometheus import create_worker_registry

    _WORKER_REGISTRY, _METRICS = create_worker_registry("xgboost")
    TRAIN_CURRENT_EPOCH = _METRICS["current_epoch"]
    TRAIN_EPOCH_DURATION_LAST = _METRICS["epoch_duration"]
    TRAIN_LOSS = _METRICS["loss"]
    TRAIN_ACCURACY = _METRICS["accuracy"]
    TRAIN_F1 = _METRICS["f1"]
    TRAIN_PRECISION = _METRICS["precision"]
    TRAIN_RECALL = _METRICS["recall"]

    _PROM_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _PROM_AVAILABLE = False
    _WORKER_REGISTRY = None
    TRAIN_CURRENT_EPOCH = None
    TRAIN_EPOCH_DURATION_LAST = None
    TRAIN_LOSS = None
    TRAIN_ACCURACY = None
    TRAIN_F1 = None
    TRAIN_PRECISION = None
    TRAIN_RECALL = None
    print(f"[xgboost_utils] Prometheus setup failed: {e}")

class RayShardIter(xgboost.core.DataIter):
    """Streams batches from a Ray Dataset shard into XGBoost without full materialization.

    Each call to ``next()`` yields one batch. XGBoost's ``QuantileDMatrix``
    constructs the internal data structure incrementally, avoiding the
    3x memory duplication of ``materialize().to_pandas()`` + ``DMatrix``.
    """

    def __init__(
        self,
        shard,
        *,
        target: str,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 4096,
    ):
        super().__init__(cache_prefix=None)  # No disk caching
        self._shard = shard
        self._target = target
        self._feature_columns = feature_columns
        self._batch_size = batch_size
        self._iter = None

    def next(self, input_data):
        if self._iter is None:
            self._iter = self._shard.iter_batches(
                batch_size=self._batch_size,
                batch_format="pandas",
                prefetch_batches=2,
            )
        try:
            batch = next(self._iter)
        except StopIteration:
            return 0  # No more data

        y = batch[self._target].to_numpy()
        if self._feature_columns:
            X = batch[self._feature_columns].to_numpy()
        else:
            X = (
                batch.drop(columns=[self._target], errors="ignore")
                .select_dtypes(include=[np.number, "bool"])
                .to_numpy()
            )

        input_data(data=X, label=y)
        return 1  # More data available

    def reset(self):
        self._iter = None


def get_train_val_dmatrix(
    target: str,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[xgboost.QuantileDMatrix, xgboost.QuantileDMatrix]:
    """Build DMatrix objects from Ray dataset shards.

    Uses ``QuantileDMatrix`` with a streaming iterator to avoid full
    materialization.  This reduces peak memory from ~6x dataset size to
    ~2x, enabling training on datasets that would otherwise OOM.

    Requires xgboost >= 2.0 for ``QuantileDMatrix``.
    """
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")
    cpus = int(os.getenv("OMP_NUM_THREADS", "1"))
    batch_size = int(os.getenv("XGBOOST_DMATRIX_BATCH_SIZE", "4096"))

    try:
        # Streaming QuantileDMatrix — avoids full materialization
        train_iter = RayShardIter(
            train_shard, target=target,
            feature_columns=feature_columns, batch_size=batch_size,
        )
        dtrain = xgboost.QuantileDMatrix(train_iter, nthread=cpus)

        # Validation: use ref=dtrain for consistent quantile bin boundaries
        val_iter = RayShardIter(
            val_shard, target=target,
            feature_columns=feature_columns, batch_size=batch_size,
        )
        dval = xgboost.QuantileDMatrix(val_iter, ref=dtrain, nthread=cpus)

        return dtrain, dval

    except Exception as e:
        raise RuntimeError(
            f"QuantileDMatrix is required for memory-efficient XGBoost training "
            f"but failed to initialize: {e}. "
            f"Ensure xgboost>=2.0 is installed."
        ) from e


def run_xgboost_train(
    *,
    params: dict,
    dtrain: xgboost.DMatrix,
    dval: xgboost.DMatrix,
    callbacks: list,
    num_boost_round: int,
    xgb_model=None,
    is_tuning: bool = False,
):
    # Start Prometheus metrics server in ALL worker processes.
    # Each worker has its own registry, so no conflicts.
    # DISABLED during tuning to avoid contaminating dashboards with trial metrics.
    world_rank = ray.train.get_context().get_world_rank()
    should_start_prometheus = (
        _PROM_AVAILABLE
        and not is_tuning  # ← NO iniciar servidor durante tuning
        and os.getenv("PROMETHEUS_ENABLE_WORKER", "1") in ("1", "true", "True")
    )
    
    if should_start_prometheus:
        try:
            port = int(os.getenv("PROMETHEUS_PORT", "8002"))
            start_http_server(port, registry=_WORKER_REGISTRY)
            print(f"[xgboost_utils] ✓ Prometheus HTTP server started on port {port} (worker registry)")
        except Exception as e:
            print(f"[xgboost_utils] Prometheus server start failed (likely already bound): {e}")
    elif is_tuning:
        print(f"[xgboost_utils] Skipping Prometheus server (tuning mode)")
    
    return xgboost.train(
        params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        verbose_eval=False,
        xgb_model=xgb_model,
        callbacks=callbacks,
    )



class RayTrainPeriodicReportCheckpointCallback(xgboost.callback.TrainingCallback):
    """Periodic metric reporting + checkpointing for Ray Train.

    Reporta métricas cada `report_every` iteraciones y checkpoint cada
    `checkpoint_every` iteraciones (solo rank 0), más un checkpoint final.
    Acepta `metric` (o `metrics`) como lista de strings en forma:
      - "validation-mlogloss"  (dataset-metric tal como XGBoost lo produce)
      - "mlogloss"             (se asume dataset "validation")
    """

    def __init__(
        self,
        *,
        report_every: int = 5,
        checkpoint_every: int = 50,
        filename: str = "model.ubj",
        metrics: Optional[List[str]] = None,
        is_tuning: bool = False,
        dval: xgboost.DMatrix = None,  # For computing accuracy/F1/etc.
        task_type: str = "classification",
    ):
        self.report_every = max(int(report_every), 1)
        self.checkpoint_every = max(int(checkpoint_every), 1)
        self.filename = filename
        self.is_tuning = is_tuning
        self.dval = dval
        self.task_type = task_type

        # Normalizamos alias: metrics override metric if ambos provistos
        if metrics is not None:
            self.metrics = list(metrics)
        else:
            # None significa "reporta todo" (comportamiento similar al oficial)
            self.metrics = None

        self._last_checkpoint_iter: Optional[int] = None
        self._last_report_dict: Dict[str, float] = {}
        self._iteration_start_time: Optional[float] = None

    def _latest_metric(self, evals_log, dataset: str, metric: str):
        try:
            v = evals_log[dataset][metric]
            return v[-1] if isinstance(v, list) else v
        except Exception:
            return None

    def _build_report_dict(self, evals_log) -> Dict[str, float]:
        report: Dict[str, float] = {}

        # Si no se especificaron métricas, reportamos todo (like Ray official)
        if self.metrics is None:
            for dataset, metrics in (evals_log or {}).items():
                for name, values in metrics.items():
                    try:
                        report[f"{dataset}-{name}"] = float(values[-1])
                    except Exception:
                        # ignoramos valores que no podamos convertir
                        pass
            return report

        # Si el usuario pasó una lista, la tratamos
        for spec in self.metrics:
            if not isinstance(spec, str):
                continue
            if "-" in spec:
                # Formato dataset-metric (ej: "validation-mlogloss")
                dataset, metric = spec.split("-", 1)
                key = spec
            else:
                # Solo metric (ej: "mlogloss") -> asumimos "validation"
                dataset = "validation"
                metric = spec
                key = f"{dataset}-{metric}"

            val = self._latest_metric(evals_log, dataset, metric)
            if val is not None:
                try:
                    report[key] = float(val)
                except Exception:
                    # si no convertible a float, lo saltamos
                    pass

        return report

    def _report(self, report_dict: Dict, model: xgboost.Booster, *, checkpoint: bool) -> None:
        world_rank = ray.train.get_context().get_world_rank()
        if checkpoint and world_rank in (0, None):
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_model(os.path.join(tmpdir, self.filename))
                ray_checkpoint = Checkpoint.from_directory(tmpdir)
                ray.train.report(report_dict, checkpoint=ray_checkpoint)
        else:
            ray.train.report(report_dict)

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        import time
        # XGBoost counts epochs from 0.
        it = epoch + 1
        
        # Track iteration duration
        current_time = time.perf_counter()
        iteration_duration = None
        if self._iteration_start_time is not None:
            iteration_duration = current_time - self._iteration_start_time
        self._iteration_start_time = current_time
        
        if it % self.report_every != 0:
            return False

        report_dict: Dict[str, float] = self._build_report_dict(evals_log)
        # Añadimos training_iteration por compatibilidad con Ray dashboards
        report_dict["training_iteration"] = it

        # Persist last metrics so `after_training` can attach them to the final
        # checkpoint report (Tune schedulers may require the metric every time).
        self._last_report_dict = dict(report_dict)

        # Prometheus: publish real-time iteration metrics from RANK 0 ONLY
        # to avoid duplicate lines in Grafana
        # DISABLED during tuning to avoid contaminating dashboards
        world_rank = ray.train.get_context().get_world_rank()
        should_export_prom = _PROM_AVAILABLE and not self.is_tuning and world_rank in (0, None)
        
        if should_export_prom:
            try:
                TRAIN_CURRENT_EPOCH.labels(framework="xgboost").set(it)
            except Exception as e:
                print(f"[xgboost_utils] Failed to export epoch: {e}")
            # Export iteration duration
            if iteration_duration is not None:
                try:
                    TRAIN_EPOCH_DURATION_LAST.labels(framework="xgboost").set(iteration_duration)
                except Exception:
                    pass
            # Export train loss (mlogloss) if available
            if "train-mlogloss" in report_dict:
                try:
                    loss_train = float(report_dict["train-mlogloss"])
                    TRAIN_LOSS.labels(framework="xgboost", split="train").set(loss_train)
                except Exception:
                    pass
            # Export validation loss (mlogloss)
            if "validation-mlogloss" in report_dict:
                try:
                    loss_val = float(report_dict["validation-mlogloss"])
                    TRAIN_LOSS.labels(framework="xgboost", split="val").set(loss_val)
                except Exception:
                    pass
            
            # Compute and export task-specific metrics on validation set
            if self.dval is not None and self.task_type == "classification":
                try:
                    y_true = self.dval.get_label().astype(int)
                    y_pred_proba = model.predict(self.dval)
                    if len(y_pred_proba.shape) > 1:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:
                        y_pred = (y_pred_proba > 0.5).astype(int)

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

                    TRAIN_ACCURACY.labels(framework="xgboost", split="val").set(acc)
                    TRAIN_PRECISION.labels(framework="xgboost", split="val").set(prec)
                    TRAIN_RECALL.labels(framework="xgboost", split="val").set(rec)
                    TRAIN_F1.labels(framework="xgboost", split="val").set(f1)

                    print(f"[xgboost_utils] Epoch {it}: acc={acc:.4f}, f1={f1:.4f}")
                except Exception as e:
                    print(f"[xgboost_utils] Failed to compute classification metrics: {e}")

        do_ckpt = (it % self.checkpoint_every == 0)
        if do_ckpt:
            self._last_checkpoint_iter = epoch
        self._report(report_dict, model, checkpoint=do_ckpt)
        return False

    def after_training(self, model):
        # Avoid duplicate checkpoint if we checkpointed on the last iteration.
        try:
            last_iter = model.num_boosted_rounds() - 1
        except Exception:
            last_iter = None

        if last_iter is not None and self._last_checkpoint_iter == last_iter:
            return model

        # Final report+checkpoint.
        # IMPORTANT: Do not report an empty dict here.
        # Ray Tune schedulers (e.g. ASHA) can run with strict metric checking and
        # will error if a result event is missing the target metric.
        final_report: Dict[str, float] = dict(self._last_report_dict) if self._last_report_dict else {}
        if "training_iteration" not in final_report:
            try:
                final_report["training_iteration"] = int(model.num_boosted_rounds())
            except Exception:
                pass

        self._report(final_report, model, checkpoint=True)
        
        # Clear Prometheus metrics after training to avoid "stuck" values during grace period
        world_rank = ray.train.get_context().get_world_rank()
        if _PROM_AVAILABLE and not self.is_tuning and world_rank in (0, None):
            try:
                # Clear all metrics so they don't show flat lines during grace period
                TRAIN_CURRENT_EPOCH.clear()
                TRAIN_EPOCH_DURATION_LAST.clear()
                TRAIN_LOSS.clear()
                TRAIN_ACCURACY.clear()
                TRAIN_F1.clear()
                TRAIN_PRECISION.clear()
                TRAIN_RECALL.clear()
                print("[xgboost_utils] Cleared Prometheus metrics after training")
            except Exception as e:
                print(f"[xgboost_utils] Failed to clear Prometheus metrics: {e}")
        
        return model
# ──────────────────────────────────────────────
# Distributed Training Function (Worker side)
# ──────────────────────────────────────────────

def train_func(config: Dict):
    """Distributed training loop for XGBoost workers.

    Used by both training and tuning pipelines.
    """
    task_type = config.get("task_type", "classification")
    task_config = get_task_config(task_type)

    params = dict(config.get("xgboost_params", XGBOOST_PARAMS))
    target = config["target"]

    # Inject task-specific objective and eval_metric if not overridden by user
    if "objective" not in params:
        params["objective"] = task_config.xgb_objective
    if "eval_metric" not in params:
        params["eval_metric"] = task_config.xgb_eval_metric

    # num_class only for classification objectives
    if task_type == "classification":
        params["num_class"] = int(config.get("num_classes", 2))

    # Align XGBoost threading with Ray worker configuration
    cpus_per_worker = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    cpus_per_worker = max(cpus_per_worker, 1)
    params["nthread"] = cpus_per_worker

    # num_boost_round is passed to xgb.train, not inside params dict
    num_boost_round = int(params.pop("num_boost_round", 100))
    if "num_boost_round" in config:
        num_boost_round = int(config["num_boost_round"])

    feature_columns = config.get("feature_columns")
    dtrain, dval = get_train_val_dmatrix(target, feature_columns=feature_columns)

    is_tuning = config.get("is_tuning", False)
    
    # Checkpoint configuration
    checkpoint_filename = "model.ubj"
    
    start_time = time.perf_counter()
    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        is_tuning=is_tuning,
        callbacks=[
            RayTrainPeriodicReportCheckpointCallback(
                metrics=["validation-mlogloss", "validation-merror", "train-mlogloss"],
                report_every=int(os.getenv("REPORT_FREQUENCY", "5")),
                checkpoint_every=int(os.getenv("RAY_TRAIN_CHECKPOINT_EVERY", "50")),
                filename=checkpoint_filename,
                is_tuning=is_tuning,
                dval=dval,
                task_type=task_type,
            )
        ],
    )
    duration = time.perf_counter() - start_time
    logger.info(f"[xgboost-worker] World rank {ray.train.get_context().get_world_rank()} finished in {duration:.2f}s")
