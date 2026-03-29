# src/ — Routing

## If the task is...

- Add a new DSL transformer (stateless) → `src/dsl/transformers.py` + `src/dsl/numpy_executor.py` (mirror) + `src/dsl/state_registry.py` (register)
- Add a new DSL estimator (learned) → `src/dsl/estimators.py` + `src/dsl/numpy_executor.py` (mirror) + `src/dsl/state_registry.py`
- Change pipeline orchestration (fit/transform order) → `src/dsl/pipeline.py`
- Change how features are selected from pipeline output → `src/dsl/base.py:PipelineModel.select_features()`
- Change online inference payload handling → `src/serve/runtime.py:ModelRuntime.predict()` + `_build_matrix()`
- Change how models are loaded from MLflow → `src/serve/registry.py:MLflowRegistry.load_by_alias()`
- Change XGBoost or PyTorch inference → `src/serve/adapters.py`
- Change canary traffic split → `src/serve/router.py:TrafficRouter`
- Change serving configuration loading → `src/serve/config.py:ConfigLoader.load()`
- Change how DSL artifacts are resolved (MLflow → Iceberg → S3) → `src/serve/pipeline_loader.py`
- Change Spark Kafka event schema → `src/converters/spark_kafka_helper.py`
- Change Python raw event conversion → `src/converters/raw_to_features.py`
- Add a new GPU provider → `src/services/gpu_catalog.py` (add `_query_{provider}()` method)
- Change GPU ranking/scoring → `src/services/gpu_selector.py`
- Change SkyPilot catalog subprocess → `src/services/_sky_catalog_query.py`
- Change training orchestration (Ray Train) → `src/pipeline/base_trainer.py`
- Change PyTorch training loop → `src/pipeline/train/pytorch.py`
- Change XGBoost training → `src/pipeline/train/xgboost.py`
- Change hyperparameter tuning → `src/pipeline/base_tuner.py` or `src/pipeline/tuning/`
- Change MLflow logging from training → `src/pipeline/utils/mlflow_utils.py`
