import time
from typing import Dict

import os
import json
import logging
import tempfile
import ray
import ray.train
from ray.train import Checkpoint
import torch
from torch import nn

# Prometheus (optional; enabled by default for training observability)
try:  # pragma: no cover
    from prometheus_client import start_http_server
    from src.prometheus import create_worker_registry

    _WORKER_REGISTRY, _METRICS = create_worker_registry("pytorch")
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
    print(f"[pytorch_utils] Prometheus setup failed: {e}")

from models.registry import get_model
from pipeline.tasks import get_task_config
from pipeline.utils.metrics_utils import regression_metrics_np

logger = logging.getLogger(__name__)


def _metrics_from_confusion(conf: torch.Tensor, *, prefix: str = "val") -> Dict[str, float]:
    # conf shape: [C, C] where rows=true, cols=pred
    conf = conf.to(torch.float32)
    support = conf.sum(dim=1)
    tp = torch.diag(conf)
    pred_sum = conf.sum(dim=0)

    precision = tp / torch.clamp(pred_sum, min=1.0)
    recall = tp / torch.clamp(support, min=1.0)
    f1 = (2.0 * precision * recall) / torch.clamp(precision + recall, min=1e-12)

    accuracy = tp.sum() / torch.clamp(conf.sum(), min=1.0)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weights = support / torch.clamp(support.sum(), min=1.0)
    weighted_precision = (precision * weights).sum()
    weighted_recall = (recall * weights).sum()
    weighted_f1 = (f1 * weights).sum()

    metrics: Dict[str, float] = {
        f"{prefix}_accuracy": float(accuracy.item()),
        f"{prefix}_precision_macro": float(macro_precision.item()),
        f"{prefix}_recall_macro": float(macro_recall.item()),
        f"{prefix}_f1_macro": float(macro_f1.item()),
        # User aliases for easier interpretation
        f"{prefix}_precision_avg": float(macro_precision.item()),
        f"{prefix}_recall_avg": float(macro_recall.item()),
        f"{prefix}_f1_avg": float(macro_f1.item()),
        f"{prefix}_precision_weighted": float(weighted_precision.item()),
        f"{prefix}_recall_weighted": float(weighted_recall.item()),
        f"{prefix}_f1_weighted": float(weighted_f1.item()),
    }

    # Per-class metrics (similar to classification_report())
    for i in range(conf.shape[0]):
        metrics[f"{prefix}_precision_class_{i}"] = float(precision[i].item())
        metrics[f"{prefix}_recall_class_{i}"] = float(recall[i].item())
        metrics[f"{prefix}_f1_class_{i}"] = float(f1[i].item())
        metrics[f"{prefix}_support_class_{i}"] = float(support[i].item())

    return metrics


def train_func(config: Dict):
    # Ray defaults to OMP_NUM_THREADS=1 unless the task/actor sets num_cpus.
    # For Ray Train workers, we pass the intended CPU budget via train_loop_config.
    cpus_per_worker = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    cpus_per_worker = max(cpus_per_worker, 1)

    torch.set_num_threads(cpus_per_worker)
    # Inter-op threads >2 often hurts on small CPU pods.
    try:
        torch.set_num_interop_threads(min(2, cpus_per_worker))
    except Exception as e:
        logger.warning(
            "[pytorch_utils] No se pudo setear torch.set_num_interop_threads(%s): %s",
            str(min(2, cpus_per_worker)),
            str(e),
            exc_info=True,
        )

    params = config["pytorch_params"]
    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 0)
    max_epochs = params.get("max_epochs", 10)
    target = config.get("target", "attack")

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    task_type = config.get("task_type", "classification")
    task_config = get_task_config(task_type)

    model_type = config.get("model_type", "mlp")
    model = get_model(model_type, {
        "input_dim": config.get("input_dim", 14),
        "num_classes": config.get("num_classes", 6),
        "d_state": params.get("d_state", 16),
        "n_estimators": params.get("n_estimators", 5),
        "latent_dim": params.get("latent_dim", 32),
    })

    loss_cls = getattr(nn, task_config.torch_loss_cls)
    loss_fn = loss_cls(**task_config.torch_loss_kwargs)

    # ── DeepSpeed (opt-in via USE_DEEPSPEED=true) ──────────────────────────────
    use_deepspeed = os.getenv("USE_DEEPSPEED", "false").lower() == "true"
    if use_deepspeed:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        ds_config = json.loads(os.getenv("DEEPSPEED_CONFIG_JSON", "{}")) or {
            "train_micro_batch_size_per_gpu": batch_size,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 1},
            "gradient_clipping": True,
            "steps_per_print": 50,
        }
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        device = get_accelerator().device_name(model.local_rank)
        print(f"[pytorch_utils] DeepSpeed ZeRO stage 1 enabled, device={device}")
    else:
        model = ray.train.torch.prepare_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ─────────────────────────────────────────────────────────────────────────────

    # Log actual CPU configuration for debugging
    world_rank = ray.train.get_context().get_world_rank()
    if world_rank == 0:
        print(f"[pytorch_utils] Worker using {cpus_per_worker} CPU thread(s) | "
              f"torch.get_num_threads()={torch.get_num_threads()}")

    # Check if we're in tuning mode (passed explicitly via config)
    is_tuning = config.get("is_tuning", False)

    # Start a Prometheus metrics server in ALL worker processes.
    # Each worker has its own registry, so no conflicts.
    # DISABLED during tuning to avoid contaminating dashboards with trial metrics.
    # We scrape the Ray worker pods directly (not the K8s submitter Job).
    # IMPORTANT: Use the worker-specific registry to avoid metric name conflicts.
    should_start_prometheus = (
        _PROM_AVAILABLE
        and not is_tuning  # ← NO iniciar servidor durante tuning
        and os.getenv("PROMETHEUS_ENABLE_WORKER", "1") in ("1", "true", "True")
    )
    
    if should_start_prometheus:
        try:
            port = int(os.getenv("PROMETHEUS_PORT", "8002"))
            # Pass the custom registry so start_http_server exposes ONLY worker metrics
            start_http_server(port, registry=_WORKER_REGISTRY)
            print(f"[pytorch_utils] ✓ Prometheus HTTP server started on port {port} (worker registry)")
        except Exception as e:
            # Best-effort: ignore if port is already in use.
            print(f"[pytorch_utils] Prometheus server start failed (likely already bound): {e}")
    elif is_tuning:
        print(f"[pytorch_utils] Skipping Prometheus server (tuning mode)")

    # Reporting cadence:
    # - For observability, default to reporting every epoch so Prometheus/Grafana
    #   can show curves evolving over time.
    # - Keep checkpoints less frequent to avoid storage overhead.
    report_every = int(os.getenv("REPORT_FREQUENCY", "5"))
    report_every = max(report_every, 1)
    checkpoint_every = int(os.getenv("RAY_TRAIN_CHECKPOINT_EVERY", "50"))
    checkpoint_every = max(checkpoint_every, 1)

    start_time = time.perf_counter()
    feature_cols = config.get("feature_columns")
    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()
        model.train()
        # prefetch_batches > 1 enables async data loading using background threads
        train_loader = train_shard.iter_torch_batches(
            batch_size=batch_size, 
            dtypes=torch.float32,
            prefetch_batches=max(2, cpus_per_worker // 2),  # Async prefetch
            local_shuffle_buffer_size=batch_size * 8,       # Randomness without high RAM
        )
        train_loss, train_batches = 0.0, 0
        for batch in train_loader:
            # Separate target from features dynamically
            if task_type == "classification":
                y = batch.pop(target).long()
            else:
                y = batch.pop(target).float()
            if feature_cols is None:
                feature_cols = sorted(batch.keys())
            X = torch.stack([batch[c] for c in feature_cols], dim=1)
            
            preds = model(X)
            loss = loss_fn(preds, y)
            if use_deepspeed:
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = train_loss / max(train_batches, 1)

        model.eval()
        val_loader = val_shard.iter_torch_batches(
            batch_size=batch_size, 
            dtypes=torch.float32,
            prefetch_batches=max(2, cpus_per_worker // 2),  # Async prefetch
        )
        val_loss, val_batches = 0.0, 0
        num_classes = int(config.get("num_classes", 2))

        # Classification: accumulate confusion matrix.  Regression: accumulate preds/targets.
        if task_type == "classification":
            conf = torch.zeros(
                (num_classes, num_classes),
                dtype=torch.int64,
                device=next(model.parameters()).device,
            )
            all_preds_list, all_targets_list = None, None
        else:
            conf = None
            all_preds_list, all_targets_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                if task_type == "classification":
                    y = batch.pop(target).long()
                else:
                    y = batch.pop(target).float()
                if feature_cols is None:
                    feature_cols = sorted(batch.keys())
                X = torch.stack([batch[c] for c in feature_cols], dim=1)

                preds = model(X)
                loss = loss_fn(preds, y)
                val_loss += loss.item()
                val_batches += 1

                if task_type == "classification":
                    y_pred = preds.argmax(dim=1)
                    # Vectorized confusion matrix via bincount
                    indices = y * num_classes + y_pred
                    counts = torch.bincount(indices, minlength=num_classes * num_classes)
                    conf += counts.reshape(num_classes, num_classes).to(conf.device)
                else:
                    y_pred = preds.squeeze(-1)
                    all_preds_list.append(y_pred.cpu())
                    all_targets_list.append(y.cpu())

        train_time_sec = time.perf_counter() - start_time
        logger.info(f"[pytorch] Worker train_time_sec={train_time_sec:.2f}")

        avg_val_loss = val_loss / max(val_batches, 1)

        epoch_time_sec = time.perf_counter() - epoch_start

        metrics_start = time.perf_counter()
        if task_type == "classification":
            metrics = _metrics_from_confusion(conf, prefix="val")
        else:
            all_p = torch.cat(all_preds_list).numpy()
            all_t = torch.cat(all_targets_list).numpy()
            metrics = regression_metrics_np(all_t, all_p, prefix="val")
        metrics["multiclass_metrics_time_sec"] = time.perf_counter() - metrics_start

        report = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time_sec,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **metrics,
        }

        should_report = ((epoch + 1) % report_every == 0) or (epoch == max_epochs - 1)
        if not should_report:
            continue

        # Prometheus: publish real-time epoch metrics from RANK 0 ONLY
        # to avoid duplicate lines in Grafana
        # DISABLED during tuning to avoid contaminating dashboards
        should_export_prom = _PROM_AVAILABLE and not is_tuning and world_rank in (0, None)
        if should_export_prom:
            try:
                TRAIN_CURRENT_EPOCH.labels(framework="pytorch").set(epoch + 1)
            except Exception:
                pass
            try:
                TRAIN_EPOCH_DURATION_LAST.labels(framework="pytorch").set(float(epoch_time_sec))
            except Exception:
                pass
            try:
                TRAIN_LOSS.labels(framework="pytorch", split="train").set(float(avg_train_loss))
            except Exception:
                pass
            try:
                TRAIN_LOSS.labels(framework="pytorch", split="val").set(float(avg_val_loss))
            except Exception:
                pass
            # Export validation performance metrics (classification only)
            if "val_accuracy" in metrics:
                try:
                    TRAIN_ACCURACY.labels(framework="pytorch", split="val").set(float(metrics["val_accuracy"]))
                except Exception:
                    pass
            if "val_f1_avg" in metrics:
                try:
                    TRAIN_F1.labels(framework="pytorch", split="val").set(float(metrics["val_f1_avg"]))
                except Exception:
                    pass
            if "val_precision_avg" in metrics:
                try:
                    TRAIN_PRECISION.labels(framework="pytorch", split="val").set(float(metrics["val_precision_avg"]))
                except Exception:
                    pass
            if "val_recall_avg" in metrics:
                try:
                    TRAIN_RECALL.labels(framework="pytorch", split="val").set(float(metrics["val_recall_avg"]))
                except Exception:
                    pass

        should_checkpoint = ((epoch + 1) % checkpoint_every == 0) or (epoch == max_epochs - 1)

        # Checkpoint: only rank 0 uploads to avoid duplicates
        if should_checkpoint and world_rank in (0, None):
            with tempfile.TemporaryDirectory() as tmpdir:
                if use_deepspeed:
                    # DeepSpeed manages optimizer state internally
                    model.save_checkpoint(tmpdir)
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                else:
                    base_model = model.module if hasattr(model, "module") else model
                    torch.save(
                        {"model_state_dict": base_model.state_dict()},
                        os.path.join(tmpdir, "model.pt"),
                    )
                    torch.save(
                        {"optimizer_state_dict": optimizer.state_dict()},
                        os.path.join(tmpdir, "optimizer.pt"),
                    )
                with open(os.path.join(tmpdir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "max_epochs": max_epochs,
                            "report_every": report_every,
                            "checkpoint_every": checkpoint_every,
                        },
                        f,
                    )
                checkpoint = Checkpoint.from_directory(tmpdir)
                ray.train.report(report, checkpoint=checkpoint)
        else:
            ray.train.report(report)
    
    # Clear Prometheus metrics after training to avoid "stuck" values during grace period
    if _PROM_AVAILABLE and not is_tuning and world_rank in (0, None):
        try:
            # Clear all metrics so they don't show flat lines during grace period
            TRAIN_CURRENT_EPOCH.clear()
            TRAIN_EPOCH_DURATION_LAST.clear()
            TRAIN_LOSS.clear()
            TRAIN_ACCURACY.clear()
            TRAIN_F1.clear()
            TRAIN_PRECISION.clear()
            TRAIN_RECALL.clear()
            print("[pytorch_utils] Cleared Prometheus metrics after training")
        except Exception as e:
            print(f"[pytorch_utils] Failed to clear Prometheus metrics: {e}")
