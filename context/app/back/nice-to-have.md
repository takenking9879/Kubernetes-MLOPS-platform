# app/back/ — Nice-to-Have

- **`GET /api/v2/runs/{id}/params`** — retrieve `params_training.yaml`; enables reproducibility in UI
- **`GET /api/v2/serving-configs`** — list all serving configs from S3; enables deployment history
- **`GET /api/v2/jobs/{id}/logs`** — proxy Airflow task logs via streaming response; removes need for Airflow UI access
- **Normalize DAG state strings** — return `{RUNNING, SUCCEEDED, FAILED}` enum instead of raw Airflow/Spark states; reduces fragile string checks in frontend
- **Unified training submission** — merge `runs.py` + `jobs.py` training paths to reduce duplication
- **Dynamic LLM catalog** — read `_LLM_CATALOG` from S3 or config file instead of hardcoding in `gpu_resources.py`
- **Authentication middleware** — even a simple `X-API-Key` header validation before production exposure
- **`GET /api/v2/datasets/{name}/stats`** — column statistics via pyiceberg table scan; useful for data exploration
