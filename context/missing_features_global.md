# Global Missing Features

## UI / Frontend

| Feature | Status | Notes |
|---------|--------|-------|
| Job cancellation button | API exists, no UI | `cancelJob()` in platformClient.ts; no page exposes it |
| Custom architecture upload workflow | API exists, partially wired | `LaunchWizardPage` lists architectures but no full upload+select flow |
| Serving config history list | No endpoint | `GET /api/v2/serving-configs` not implemented; no "past deployments" view |
| Training run params viewer | No endpoint | No `GET /api/v2/runs/{id}/params`; users can't inspect historical params |
| Dataset preview | API exists, not used | `getIcebergSample()` defined but no "Preview Data" page |
| Artifact validation before training | API exists, not called | `checkArtifact()` could validate preprocess_run_id; never invoked |
| DAG / task log viewer | Missing entirely | No backend endpoint for streaming Airflow task logs |
| Model performance monitoring | Missing | No serving inference metrics / drift detection UI |
| Dataset delete / rename | Missing | Only create, list, upload available |
| Error state persistence | Missing | Errors disappear on page navigation |

## Backend

| Feature | Status | Notes |
|---------|--------|-------|
| `GET /api/v2/runs/{id}/params` | Missing | Training params YAML not retrievable after submission |
| `GET /api/v2/serving-configs` | Missing | No list endpoint for serving configs |
| `GET /api/v2/jobs/{id}/logs` | Missing | No endpoint to fetch Airflow/SkyPilot task logs |
| `DELETE /api/v2/datasets/{name}` | Missing | No dataset deletion |
| Cost tracking / billing | Missing | No historical cost or budget alerts endpoint |
| Authentication / authorization | Missing | No API key or JWT validation; relies on ingress isolation |

## src/ Core Logic

| Feature | Status | Notes |
|---------|--------|-------|
| BAE / SSM training integration | Missing | `src/models/bae.py`, `ssm.py` exist but no training pipeline uses them |
| Request payload validation | Minimal | No JSON schema validation in serving `predict()`; only basic type checks |
| GPU rate limiting | Missing | No rate limiting on external GPU API calls; relies on TTL cache |
| Model explainability | Missing | No SHAP/LIME wrappers in serving adapters |
| Drift detection | Missing | Synthetic data generator supports concept/data drift but no detector in serving |
| Batch prediction API | Missing | No separate batch inference endpoint (only streaming Kafka) |
| Ray cluster auto-scaling | Unknown | How Ray worker pools scale with GPU availability not documented |
| Artifact cleanup | Missing | No S3 cleanup for old `stages.json` / pipeline artifacts |

## k3s/ Orchestration

| Feature | Status | Notes |
|---------|--------|-------|
| Serving config list/history | Missing | No DAG to list or archive past serving configs |
| Spark connector cleanup | Missing | No mechanism to stop/replace running Kafka connector on redeploy |
| Monitoring DAGs | Missing | No automated retraining / drift-triggered retraining DAG |
| GPU catalog auto-refresh in Airflow | Missing | GPU catalog is query-time only; no pre-warming or scheduled refresh |
