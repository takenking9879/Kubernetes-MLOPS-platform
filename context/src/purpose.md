# src/ — Purpose

### Purpose
Core logic for the platform. Contains:
- **DSL**: Spark-based declarative feature engineering (YAML-configured transformers + estimators)
- **NumPy executor**: Pure-Python online inference version of the DSL (no Spark dependency)
- **Serving runtime**: Ray Serve-based prediction with canary routing and MLflow alias loading
- **Converters**: Raw Kafka event → feature dict (Spark and Python versions)
- **GPU services**: Real-time GPU catalog (RunPod, Vast.ai, AWS) and SkyPilot selector
- **Training pipeline**: Distributed Ray Train (PyTorch + XGBoost) with Ray Tune HPO

### When to use
- Modifying feature engineering transforms (Spark or online) → `src/dsl/`
- Modifying inference/prediction logic → `src/serve/`
- Modifying GPU catalog or pricing → `src/services/`
- Modifying training/tuning logic → `src/pipeline/`
- Modifying raw event to feature conversion → `src/converters/`

### When not to use
- API endpoints → `app/backend/routers/`
- UI → `app/frontend/`
- Airflow DAG logic → `k3s/airflow/dags/`
- K8s manifests → `k3s/`

### Physical layout
```
src/
  dsl/                  ← Spark DSL pipeline (transformers, estimators, pipeline, numpy mirror)
  serve/                ← Ray Serve runtime (ModelRuntime, registry, adapters, canary)
  converters/           ← Kafka event converters (Spark + Python)
  services/             ← GPU catalog, selector, SkyPilot subprocess helper
  pipeline/             ← Ray Train distributed training + Ray Tune HPO
  models/               ← Model wrappers (PyTorch, XGBoost, BAE, SSM)
  schemas/              ← Data schemas and hyperparameter schemas
  prometheus/           ← Monitoring + evaluation utilities
  calculator/           ← Resource cost estimation
  iceberg/              ← Iceberg metadata query utilities
  utils/                ← Logger, base class
```
