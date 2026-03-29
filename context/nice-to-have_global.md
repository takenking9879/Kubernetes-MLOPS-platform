# Nice-to-Have (Global)

## UX / Frontend

- **Job cancellation UI** — expose `cancelJob()` as a button in status polling pages; highest-impact quick win
- **Dataset preview page** — use `getIcebergSample()` (already defined); show column stats + first N rows
- **Training params viewer** — add `GET /api/v2/runs/{id}/params`; allow users to inspect historical runs
- **Error toasts / global error state** — currently errors disappear on navigation; add persistent error boundary
- **Polling progress bars** — replace text-only polling with step indicators (Spark → complete → training → complete)
- **Cost estimate timeline** — show `estimated_cost_spot * runtime_hours` before job submission

## Backend

- **`GET /api/v2/serving-configs`** — list all serving configs from S3 prefix; enables deployment history
- **`GET /api/v2/runs/{id}/params`** — retrieve `params_training.yaml` content; enables reproducibility view
- **`GET /api/v2/jobs/{id}/logs`** — proxy Airflow task logs to frontend; replaces need to log into Airflow UI
- **Dataset stats endpoint** — `GET /api/v2/datasets/{name}/stats` using Iceberg table metadata
- **Authentication layer** — even a simple API key header check for production hardening

## src/ Core Logic

- **BAE / SSM training integration** — `src/models/bae.py`, `ssm.py` are present; wire into `src/pipeline/train/`
- **Request payload schema validation** — add Pydantic model for `predict()` input; cleaner error messages
- **`calculator/` integration** — tie `resource_calculator.py` cost estimates into `GPUSelectorService` for pre-job cost projection
- **SHAP explanations** — add optional SHAP values to `XGBoostAdapter.predict()` response
- **Drift detection** — lightweight feature distribution tracker in `ModelRuntime`; emit Prometheus metric on drift

## k3s/ Orchestration

- **Spark Kafka connector lifecycle DAG** — add `stop_kafka_connector` task or separate DAG for cleanup
- **Retraining trigger DAG** — listen to Prometheus drift alert → trigger `preprocessing_pipeline` + `training_pipeline_skypilot`
- **GPU catalog warm-up** — scheduled 5-min DAG to call `gpu_catalog.py` and cache results before user requests
- **Single generic legacy file cleanup** — delete `dag.py`, `hello-sky.yaml`, `vast-test.yaml`, `ray-gpu-training.yaml` (generic superseded YAMLs), `params.yaml`, `kuberay-job-gpu.yaml`; see `context/k3s/legacy.md`

## Architecture

- **Unified job status** — normalize all DAG states (Spark "ResourceReleased", Ray "succeeded", SkyPilot "SUCCEEDED") into a single `{RUNNING, SUCCEEDED, FAILED}` enum in backend; reduces fragile string checks in frontend
- **GCP / Azure GPU providers** — `gpu_catalog.py` supports adding new providers; RunPod+Vast+AWS is current coverage
- **Feature Store** — integrate Iceberg as first-class feature store (materialize features, not just store processed tables)
