# src/ — Mismatches

- **`src/models/bae.py`, `ssm.py`**: Present and referenced in `routers/config.py` as supported types, but no training pipeline integration. Either integrate or remove from config manifest.

- **`src/prometheus/preprocessing.py`**: References metrics but not integrated into training pipeline metrics exports. `src/serve/runtime.py` has its own Prometheus counters. Two parallel metric systems.

- **`src/calculator/`**: Cost estimation logic exists but disconnected from GPU selection. GPUSelectorService doesn't use it; frontend estimates cost independently.

- **`src/serve/webhooks.py`**, **`xgboost.py`**, **`pytorch.py`**: Stub files; actual logic lives in `registry.py` and `adapters.py`. Risk of confusion for future developers.

- **`tasks.py` (Celery)**: Present but minimal; actual training goes through Ray Train. Celery not in the critical path.

- **Config loading 3-way merge**: `src/serve/config.py:ConfigLoader` merges from 3 sources. If a new params YAML key is added, it must be handled in the merge logic — otherwise defaults silently take precedence.

- **DSL YAML vs NumPy executor sync**: Any new DSL stage added to `transformers.py` or `estimators.py` MUST also be added to `numpy_executor.py`. No automated check enforces this.
