# app/back/ — Routing

## If the task is...

- Add a new endpoint → find domain router in `routers/`; if new domain, create file + mount in `main.py`
- Change dataset upload or ingest → `routers/datasets.py`
- Change DSL versioning or storage → `routers/dsls.py`
- Change preprocessing run trigger → `routers/processing_runs.py`
- Change training run trigger → `routers/runs.py`
- Change schema versioning → `routers/schemas.py`
- Change serving config or deploy → `routers/serving_configs.py`
- Change vLLM deploy → `routers/serving_configs.py` (`/vllm-deploy`)
- Change training capabilities manifest → `routers/config.py`
- Change GPU catalog API or response fields → `routers/gpu_resources.py` + `_offer_to_dict()`
- Change GPU provider data source → `src/services/gpu_catalog.py`
- Change SkyPilot selection/ranking → `src/services/gpu_selector.py`
- Change job launch orchestration routing → `services/orchestration_selector.py`
- Change SkyPilot YAML generation → `services/job_builder.py`
- Change custom model upload validation → `routers/model_architectures.py`
- Change CORS, startup, middleware → `main.py`
- Change static training config (frameworks, metrics) → `routers/config.py`
