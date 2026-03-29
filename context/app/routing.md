# app/ — Routing

## If the task is...

- Change what happens when user clicks a button → `app/frontend/src/pages/{Page}.tsx`
- Change what data a component displays → `app/frontend/src/components/{Component}.tsx`
- Add a new API call from frontend → `app/frontend/src/api/platformClient.ts`
- Add/change a backend endpoint → `app/backend/routers/{domain}.py`
- Change what external service a router calls → `app/backend/routers/{domain}.py` or `app/backend/services/`
- Change the FastAPI app startup / CORS / router mounting → `app/backend/main.py`
- Change Docker build → `app/Dockerfile`

## Page → router mapping

| Page | API calls | Backend router |
|------|-----------|----------------|
| DatasetPage | listDatasets, createDataset, uploadParquets, submitIngest, getIngestStatus | `routers/datasets.py` |
| DSL Builder | listDsls, getDsl, saveDsl, loadSchemaFromIceberg | `routers/dsls.py`, `main.py` |
| ProcessingPage | submitProcessingRun, uploadFullSchema, getProcessingRunStatus | `routers/processing_runs.py`, `routers/schemas.py` |
| RunPage | listPreprocessRunIds, getPreprocessParams, submitRun, getRunStatus | `routers/processing_runs.py`, `routers/runs.py`, `routers/config.py` |
| ServingPage | listTrainingRunIds, submitServingConfig, triggerServingDeploy, triggerVllmDeploy | `routers/runs.py`, `routers/serving_configs.py`, `routers/gpu_resources.py` |
| LaunchWizardPage | launchJob, listArchitectures, uploadArchitecture, getLLMCatalog | `routers/jobs.py`, `routers/model_architectures.py`, `routers/gpu_resources.py` |
