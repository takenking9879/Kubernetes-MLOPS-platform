# Global Mismatches

## front ↔ back

| Mismatch | Detail | Impact |
|----------|--------|--------|
| `checkArtifact()` unused | Defined in `platformClient.ts`, backend endpoint exists (`GET /api/v2/runs/check`), but no page calls it | Could validate preprocess_run_id before training; currently silent |
| `uploadSchemas()` (batch) unused | `POST /api/v2/datasets/{name}/schemas` exists, but pages use individual `uploadFullSchema/uploadRawSchema` | Both endpoints exist in parallel; legacy one is dead |
| `listProcessingRuns()` unused | `GET /api/v2/processing-runs` exists, but pages use `/ids` endpoint instead | Redundant endpoint |
| Model architecture upload incomplete | Backend fully supports upload + AST validation; `LaunchWizardPage` calls `listArchitectures()` but doesn't wire the upload workflow | Feature half-built |
| No job cancellation UI | `cancelJob()` exists in `platformClient.ts` and backend; no page exposes a cancel button | Can't cancel running jobs from UI |

## back ↔ k3s/

| Mismatch | Detail | Impact |
|----------|--------|--------|
| `training_pipeline` vs `training_pipeline_skypilot` | Backend RunPage submits with `skypilot` query param routing to different DAGs; LaunchWizardPage always uses `training_pipeline_skypilot` | Two parallel training paths; must keep both DAGs |
| Missing `vllm-deploy` polling display | Backend stores endpoint in S3 after `vllm_serving_pipeline` completes; frontend polls `getVllmEndpoint()` but status display is minimal | Endpoint retrieval works but UI doesn't clearly communicate status |
| `serving_pipeline` Spark Kafka connector never deleted on success | DAG intentionally leaves Spark connector running (streaming); but there's no cleanup mechanism if serving is replaced | Stale Spark connectors accumulate |

## src/ ↔ back/

| Mismatch | Detail | Impact |
|----------|--------|--------|
| `src/pipeline/` not imported by backend | Training code runs in cloud pods; params passed via YAML, not direct imports | Normal architecture, but YAML schema must stay in sync with trainer expectations |
| `src/serve/` not imported by backend | Serving code runs in RayService pod; backend only patches K8s resource | Normal, but `PARAMS_SERVING_S3_PATH` env injection must match `src/serve/config.py` load logic |
| `src/models/bae.py`, `ssm.py` | Present but no training pipeline integration found | Likely prior art or future experiments; unused |

## src/ internal mismatches

| Mismatch | Detail |
|----------|--------|
| `prometheus/preprocessing.py` | References metrics but not integrated in training pipeline metrics exports |
| `calculator/` | Resource cost estimation exists but not integrated with GPUSelectorService |
| `models/bae.py`, `ssm.py` | Defined in `back/routers/config.py` as supported types (BAE, SSM) but no training loop integration |
| `tasks.py` (Celery) | Present but minimal; actual training goes through Ray Train, not Celery |

## k3s/ internal

| Mismatch | Detail |
|----------|--------|
| `training_pipeline` vs `training_pipeline_skypilot` | Two separate training DAGs; `training_pipeline` uses KubeRay locally; `training_pipeline_skypilot` uses cloud GPU. Both are ACTIVE; duplication is intentional (local vs cloud) |
| `ml_pipeline` overlaps with `full_ml_pipeline` | Both chain preprocessing + training; `full_ml_pipeline` is for code, `ml_pipeline` is the UI-triggered flexible mode. Risk of divergence |
| RBAC for `sky-runner` pod | `sky-runner-pod.yaml` (KubernetesPodOperator) needs credentials for RunPod/Vast/AWS; env-secret pattern assumed but not verified |
