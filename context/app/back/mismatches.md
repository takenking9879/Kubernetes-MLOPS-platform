# app/back/ — Mismatches

- **`POST /api/v2/datasets/{name}/schemas` (batch)** — legacy endpoint that accepts raw+full+preprocessed in one call; no frontend page calls it. Individual `/api/v2/schemas/{dataset}/full|raw` are used instead. The batch endpoint is effectively dead.

- **`GET /api/v2/runs/check`** — implemented but never called by frontend. Returns `{exists, processed_table}` for a given `execution_id + dataset`.

- **`GET /api/v2/processing-runs`** (full list) — implemented but no frontend page calls it. Frontend uses `/ids` instead.

- **Two training DAGs** — `training_pipeline` (KubeRay local) and `training_pipeline_skypilot` (cloud). Backend `runs.py` routes based on `?skypilot=true` query param; `jobs.py` always uses `training_pipeline_skypilot`. Must keep both DAGs in sync.

- **`job_builder.py` vs `runs.py`** — training run submission exists in two places: `runs.py` (for RunPage) and `jobs.py` (for LaunchWizardPage). Different code paths for what is conceptually the same action. Risk of divergence.

- **Static LLM catalog** — `gpu_resources.py` has a hardcoded `_LLM_CATALOG` list (4 models). Not read from any config file or S3. Adding a new LLM model requires code change.

- **BAE/SSM model types in config** — `routers/config.py` exposes BAE and SSM as supported frameworks, but `src/pipeline/` has no training loop integration for these types.
