# src/ — Missing Features

- **BAE training integration** — `src/models/bae.py` exists; `routers/config.py` lists BAE as a supported type; but no training loop in `src/pipeline/` uses it
- **SSM training integration** — same situation as BAE (`src/models/ssm.py`)
- **`calculator/` integration** — `src/calculator/resource_calculator.py` estimates compute costs but is not wired into `GPUSelectorService` for pre-job cost projection
- **Drift detection** — no live drift detector in `ModelRuntime`; synthetic data generator supports drift modes but serving doesn't detect it
- **Model explainability** — no SHAP/LIME wrappers in `adapters.py`
- **Request payload validation** — `ModelRuntime.predict()` does minimal schema validation; Pydantic model would improve error messages
- **Artifact cleanup** — no mechanism to delete stale `stages.json` / pipeline artifacts from S3
- **Batch prediction API** — only streaming Kafka inference; no batch prediction endpoint
