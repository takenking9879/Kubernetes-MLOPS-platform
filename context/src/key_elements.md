# src/ — Key Elements (Top Level)

For sub-module detail see:
- `context/src/dsl/key_elements.md`
- `context/src/serve/key_elements.md`
- `context/src/services/key_elements.md`
- `context/src/converters/purpose.md`

## Module Summary

| Module | Key Files | Purpose |
|--------|-----------|---------|
| `src/dsl/` | `base.py`, `transformers.py`, `estimators.py`, `pipeline.py`, `numpy_executor.py`, `state_registry.py` | Spark DSL + online NumPy mirror |
| `src/serve/` | `runtime.py`, `config.py`, `registry.py`, `adapters.py`, `router.py`, `pipeline_loader.py` | Ray Serve inference |
| `src/converters/` | `spark_kafka_helper.py`, `raw_to_features.py` | Event → feature dict |
| `src/services/` | `gpu_catalog.py`, `gpu_selector.py`, `_sky_catalog_query.py` | GPU catalog + SkyPilot selection |
| `src/pipeline/` | `base_trainer.py`, `base_tuner.py`, `train/pytorch.py`, `train/xgboost.py` | Ray Train distributed training (PyTorch worker loop now enforces explicit tensor→model device placement and supports GPU diagnostics via `RAY_GPU_DEBUG`) |
| `src/models/` | `bae.py`, `ssm.py`, `pytorch.py`, `xgboost.py` | Model wrappers (BAE/SSM unused) |
| `src/schemas/` | `spark/schemas.py`, `model/pytorch_params.py`, `model/xgboost_params.py` | Data + hyperparameter schemas |
| `src/iceberg/` | `metadata_query_utils.py` | Resolve artifact_set_id → pipeline_hash via Iceberg |
| `src/prometheus/` | `metrics.py` | Prometheus metrics export |
| `src/calculator/` | `resources.py`, `resource_calculator.py` | Resource cost estimation (unused by GPU selector) |

## Dual Execution Pattern

Every DSL operation exists in two forms:
1. **Spark** (`src/dsl/transformers.py`, `estimators.py`) — batch training/preprocessing
2. **NumPy** (`src/dsl/numpy_executor.py`) — online inference in Ray Serve, no Spark dependency

The two must stay in sync. When adding a transformer, add it in both files.

## Model Artifact Chain (MLflow → Iceberg → S3 → NumpyExecutor)

```
ModelRuntime / PipelineArtifactLoader
  → MLflow tag "artifact_set_id" on model version
    → Iceberg: iceberg.metadata.preprocessing_artifacts (pipeline_hash)
      → S3: {bucket}/pipelines/{pipeline_hash}/stages.json + config.json
        → NumpyPipelineExecutor.from_dir()
```

File: `src/serve/pipeline_loader.py:PipelineArtifactLoader.load_executor()`
