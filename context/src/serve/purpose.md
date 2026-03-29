# src/serve/ — Purpose

### Purpose
Ray Serve-based model inference runtime. Handles two payload formats (pre-processed and raw), loads models from MLflow by alias, routes traffic between stable/canary variants, and manages webhook-synchronized alias updates.

### When to use
- Changing how prediction payloads are processed
- Changing model loading from MLflow
- Changing canary traffic split logic
- Changing serving configuration loading
- Changing DSL artifact resolution chain (MLflow → Iceberg → S3)

### When not to use
- DSL transform logic → `src/dsl/`
- Serving config API (HTTP endpoint) → `app/backend/routers/serving_configs.py`
- Serving K8s deployment → `k3s/kuberay/serving/`
- Serving DAG logic → `k3s/airflow/dags/serving_dag.py`

### Physical layout
```
src/serve/
  runtime.py          ← ModelRuntime: predict(), _build_matrix(), metrics
  config.py           ← ServingConfig, ConfigLoader (3-source merge)
  registry.py         ← MLflowRegistry: load_by_alias(), ensure_webhook()
  adapters.py         ← XGBoostAdapter, PyTorchAdapter (framework inference)
  router.py           ← TrafficRouter (stable/canary split by probability)
  pipeline_loader.py  ← PipelineArtifactLoader: MLflow→Iceberg→S3→NumpyExecutor
  webhooks.py         ← stub (logic in registry.py)
  xgboost.py          ← stub (logic in adapters.py)
  pytorch.py          ← stub (logic in adapters.py)
```
