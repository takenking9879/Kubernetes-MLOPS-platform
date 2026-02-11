import os
import json
import logging
import tempfile
import numbers
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def log_evaluation_artifacts(metrics: Dict[str, Any], framework: str):
    """
    Generates and logs confusion matrix plots and classification reports to MLflow.
    """
    # Prefer standardized names (val_*) but fallback to legacy names (*)
    cm_val = metrics.get("val_confusion_matrix") or metrics.get("confusion_matrix")
    report_val = metrics.get("val_classification_report") or metrics.get("classification_report")
    
    cm_test = metrics.get("test_confusion_matrix")
    report_test = metrics.get("test_classification_report")

    if not any([cm_val, report_val, cm_test, report_test]):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        def _log_single_eval(prefix: str, cm_obj, report_text) -> None:
            if report_text:
                report_path = os.path.join(tmpdir, f"{prefix}_classification_report.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(str(report_text))
                mlflow.log_artifact(report_path, artifact_path="evaluation")

            if cm_obj is not None:
                # Save JSON
                cm_json_path = os.path.join(tmpdir, f"{prefix}_confusion_matrix.json")
                with open(cm_json_path, "w", encoding="utf-8") as f:
                    json.dump(cm_obj, f)
                mlflow.log_artifact(cm_json_path, artifact_path="evaluation")

                # Save PNG
                cm_np = np.asarray(cm_obj)
                # For imbalanced datasets, a normalized colormap is more interpretable than raw counts.
                # We still annotate with raw counts + per-row percentages.
                row_sum = cm_np.sum(axis=1, keepdims=True).astype(np.float64)
                denom = np.where(row_sum == 0.0, 1.0, row_sum)
                cm_norm = cm_np.astype(np.float64) / denom

                scale = 0.8
                fig_w = min(14 * scale, max(6 * scale, cm_np.shape[0] * 0.75 * scale))
                fig_h = min(12 * scale, max(5 * scale, cm_np.shape[0] * 0.60 * scale))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{prefix.upper()} Confusion Matrix (row-normalized) ({framework})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_xticks(range(cm_np.shape[1]))
                ax.set_yticks(range(cm_np.shape[0]))
                ax.set_xticklabels([str(i) for i in range(cm_np.shape[1])], rotation=45, ha="right")
                ax.set_yticklabels([str(i) for i in range(cm_np.shape[0])])

                # Annotate each cell with count and percentage.
                # Choose text color based on normalized intensity.
                for i in range(cm_np.shape[0]):
                    for j in range(cm_np.shape[1]):
                        count = int(cm_np[i, j])
                        pct = float(cm_norm[i, j] * 100.0)
                        txt = f"{count}\n{pct:.1f}%"
                        color = "white" if cm_norm[i, j] >= 0.5 else "black"
                        ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
                plt.tight_layout()

                png_path = os.path.join(tmpdir, f"{prefix}_confusion_matrix.png")
                fig.savefig(png_path, dpi=160)
                plt.close(fig)
                mlflow.log_artifact(png_path, artifact_path="evaluation")

        _log_single_eval("val", cm_val, report_val)
        _log_single_eval("test", cm_test, report_test)

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_and_register_model(
    *,
    model,
    framework: str,
    registered_model_name: str,
    artifact_set_id: Optional[str] = None,
    table_identifier: Optional[str] = None,
    run_id: str,
) -> Optional[str]:
    """Log model using native MLflow flavor and register in the Model Registry.

    Returns the model version string, or None on failure.
    """
    try:
        # Log using the native MLflow flavor for best compatibility
        if framework == "xgboost":
            import mlflow.xgboost
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif framework == "pytorch":
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, artifact_path="model")
        else:
            logger.warning("Unknown framework '%s' – skipping model registration.", framework)
            return None

        # Register in the Model Registry (creates the registered model if needed)
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, registered_model_name)
        version = str(mv.version)

        # Attach traceability tags to the model *version*
        # artifact_set_id links back to Iceberg metadata which already contains
        # the split ranges, raw_snapshot_id, pipeline_hash, etc.
        client = MlflowClient()
        version_tags: Dict[str, str] = {"framework": framework}
        if artifact_set_id:
            version_tags["artifact_set_id"] = artifact_set_id
        if table_identifier:
            version_tags["iceberg_table"] = table_identifier

        for tag_key, tag_value in version_tags.items():
            client.set_model_version_tag(
                registered_model_name, version, tag_key, str(tag_value)
            )

        logger.info(
            "Model registered: %s v%s (artifact_set_id=%s)",
            registered_model_name, version, artifact_set_id,
        )
        return version

    except Exception as e:
        logger.error("Failed to register model in MLflow: %s", e, exc_info=True)
        return None


def set_model_alias(
    tracking_uri: str,
    model_name: str,
    version: str,
    alias: str,
) -> None:
    """Assign an alias to a model version (MLflow 3.x).

    MLflow 3.x replaced the Stages lifecycle (Staging/Production/Archived)
    with **aliases**: free-form strings that point to exactly one version.
    Common aliases: ``champion``, ``challenger``, ``archived``.

    An alias automatically moves away from whichever version had it before,
    so there is no need to manually unset it.

    Args:
        tracking_uri: MLflow tracking server URI.
        model_name: Registered model name in the Model Registry.
        version: Model version number (string).
        alias: Target alias, e.g. ``"champion"`` or ``"challenger"``.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(version),
    )
    logger.info("Model %s v%s aliased as '%s'", model_name, version, alias)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point for training runs
# ─────────────────────────────────────────────────────────────────────────────

def log_training_run(
    framework: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_location: str,
    *,
    # ── Traceability (data lineage) ──
    artifact_set_id: Optional[str] = None,
    table_identifier: Optional[str] = None,
    # ── Model Registry ──
    model: Any = None,
    registry_model_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Handles the full MLflow lifecycle for a *final* training run:

    1. Experiment tracking (params, metrics, evaluation artifacts).
    2. Data-lineage tags (artifact_set_id, Iceberg table).
       Split ranges are already stored in the Iceberg metadata table
       linked via ``artifact_set_id``, so we don't duplicate them here.
    3. Model logging + Model Registry registration (when *model* is provided).

    Returns a dict with ``run_id``, ``model_version``, and ``registry_model_name``
    so the caller can promote the version (e.g. via ``set_model_alias``).
    """
    tracking_uri = params.get("mlflow_tracking_uri")
    experiment_name = params.get("mlflow_experiment_name")

    if not tracking_uri or not experiment_name:
        return None

    mlflow.set_tracking_uri(tracking_uri)

    # Ensure experiment exists
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)

    mlflow.set_experiment(experiment_name)

    run_name = f"{params.get('name', framework)}_final_{framework}"
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Log Parameters
        mlflow.log_param("framework", framework)
        mlflow.log_param("target", params.get("target"))
        mlflow.log_param("num_classes", params.get("num_classes"))
        mlflow.log_param("num_workers", os.getenv("NUM_WORKERS", ""))

        if framework == "xgboost" and params.get("xgboost_params"):
            for k, v in params["xgboost_params"].items():
                mlflow.log_param(f"xgb_{k}", v)
        elif framework == "pytorch" and params.get("pytorch_params"):
            for k, v in params["pytorch_params"].items():
                mlflow.log_param(f"pt_{k}", v)

        # 2. Data-Lineage Tags
        mlflow.set_tag("training_type", "distributed_final")
        if artifact_set_id:
            mlflow.set_tag("artifact_set_id", artifact_set_id)
        if table_identifier:
            mlflow.set_tag("iceberg_table", table_identifier)

        # 3. Log Numeric Metrics
        for k, v in (metrics or {}).items():
            if isinstance(v, numbers.Real) and not isinstance(v, bool):
                mlflow.log_metric(k, float(v))

        # 4. Log Evaluation Artifacts (Confusion Matrices, Classification Reports)
        log_evaluation_artifacts(metrics, framework)

        # 5. Model Registry: log model + register a new version
        model_version = None
        effective_registry_name = None
        if model is not None:
            effective_registry_name = registry_model_name or experiment_name
            model_version = _log_and_register_model(
                model=model,
                framework=framework,
                registered_model_name=effective_registry_name,
                artifact_set_id=artifact_set_id,
                table_identifier=table_identifier,
                run_id=run.info.run_id,
            )

    return {
        "run_id": run.info.run_id,
        "model_version": model_version,
        "registry_model_name": effective_registry_name,
    }
