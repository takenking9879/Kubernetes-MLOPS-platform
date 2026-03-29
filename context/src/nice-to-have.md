# src/ — Nice-to-Have

- **BAE/SSM training pipeline** — wire `src/models/bae.py`, `ssm.py` into `src/pipeline/train/` as framework trainers; expose in `routers/config.py`
- **SHAP explanations in XGBoostAdapter** — optional `explain=True` flag in `predict()`; returns SHAP values alongside predictions
- **Drift detection in ModelRuntime** — lightweight feature distribution tracker; emit `serve_drift_detected` Prometheus metric when score exceeds threshold
- **Pydantic payload validation** — validate `predict()` input schema before processing; better error messages for malformed requests
- **`calculator/` integration** — expose `resource_calculator.estimated_cost()` in `GPUSelectResult`; used for pre-job cost warnings in UI
- **Automated sync check** — a test that verifies every stage type in `state_registry.py` has a corresponding implementation in `numpy_executor.py`
- **GCP / Azure provider** — `gpu_catalog.py` is structured for extensibility; adding `_query_gcp()` or `_query_azure()` follows the same pattern
