# app/back/ — Missing Features

- **`GET /api/v2/runs/{id}/params`** — no endpoint to retrieve `params_training.yaml` after a run; users can't inspect historical training configs
- **`GET /api/v2/serving-configs`** — no list endpoint; can only fetch individual configs by ID; no deployment history view
- **`GET /api/v2/jobs/{id}/logs`** — no endpoint to proxy Airflow task logs; users must access Airflow UI directly
- **`DELETE /api/v2/datasets/{name}`** — no dataset deletion endpoint
- **`GET /api/v2/datasets/{name}/stats`** — no profiling or column statistics endpoint
- **`GET /api/v2/model-architectures/{arch_id}`** — no single-architecture fetch; only list all
- **`DELETE /api/v2/serving-configs/{id}/deploy/{dag_run_id}`** — no serving-specific cancel endpoint (job cancel via `/api/v2/jobs/{id}` is available but not serving-specific)
- **Authentication** — no API key or JWT validation on any endpoint; relies entirely on ingress/network isolation
- **Rate limiting** — no rate limiting; GPU catalog API calls could overwhelm providers
