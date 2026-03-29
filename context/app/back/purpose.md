# app/back/ — Purpose

### Purpose
FastAPI server that bridges the React frontend with all backend systems: S3, Iceberg, Airflow, K8s, MLflow, and src/ services. Does NOT run training or serving — it only generates params, uploads configs, and triggers Airflow DAGs.

### When to use
- Adding or changing an API endpoint
- Changing what data gets written to S3 or Iceberg
- Changing how Airflow DAGs are triggered or polled
- Adding request/response validation (Pydantic models)
- Changing GPU catalog or selection logic exposed via API

### When not to use
- Feature engineering logic → `src/dsl/`
- Training logic → `src/pipeline/`
- Serving runtime → `src/serve/`
- Airflow DAG internals → `k3s/airflow/dags/`
- SkyPilot YAML content → `k3s/sky/`

### Physical layout
```
app/backend/
  main.py              ← FastAPI app; mounts all 10 routers; legacy endpoints
  routers/
    datasets.py        ← /api/v2/datasets
    dsls.py            ← /api/v2/datasets/{name}/dsls
    processing_runs.py ← /api/v2/processing-runs
    runs.py            ← /api/v2/runs
    schemas.py         ← /api/v2/schemas
    serving_configs.py ← /api/v2/serving-configs
    config.py          ← /api/v2/config
    gpu_resources.py   ← /api/v2/gpu-resources
    jobs.py            ← /api/v2/jobs
    model_architectures.py ← /api/v2/model-architectures
  services/
    orchestration_selector.py ← select DAG + YAML based on job type
    job_builder.py     ← generate params_training.yaml + SkyPilot YAML
```
