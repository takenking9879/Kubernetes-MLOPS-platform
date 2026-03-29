# Cross-Layer Relationships

## front ↔ back

All frontend communication goes through `app/frontend/src/api/platformClient.ts`.
- Every `GET/POST/DELETE` in that file maps 1:1 to a FastAPI endpoint.
- **Coverage**: All 10 backend routers have corresponding frontend API functions. No orphaned endpoints found.
- **Exception**: `GET /api/v2/runs/check` (checkArtifact) is defined in client but never called by any page.
- Frontend never directly contacts Airflow, S3, Iceberg, or MLflow — all go through backend.

## back ↔ src/

Backend routers import from `src/services/`:

| Router | Imports |
|--------|---------|
| `gpu_resources.py` | `GPUCatalogService`, `GPUSelectorService`, `ResourceConstraints` |
| `jobs.py` | `JobBuilder`, `TrainingJobConfig`, `ServingJobConfig` (via `app/backend/services/`) |
| `processing_runs.py` | pyiceberg (direct), boto3 (direct) |
| `runs.py` | boto3, pyiceberg, Airflow REST API (direct) |
| `model_architectures.py` | ast (standard lib), boto3 |

`src/pipeline/` (Ray Train) is NOT imported by backend — it runs inside the cloud GPU pod launched by SkyPilot. The backend only generates the YAML/params and triggers Airflow.

`src/serve/` is NOT imported by backend — it runs inside the KubeRay RayService pod. The backend only patches the RayService CRD via K8s API.

`src/dsl/` is NOT imported by backend at runtime — it runs inside the SparkApplication pod. The backend only uploads DSL YAMLs to S3.

## back ↔ k3s/

| Backend action | k3s artifact used |
|---------------|-------------------|
| Submit ingest | Creates SparkApplication from `k3s/spark/ingestion/spark-application-ingestion.yaml` |
| Submit processing | Triggers `preprocessing_pipeline` DAG → uses `k3s/spark/spark-application.yaml` |
| Submit training (local) | Triggers `training_pipeline` DAG → uses `k3s/kuberay/kuberay-job.yaml` |
| Submit training (SkyPilot) | Triggers `training_pipeline_skypilot` DAG → uses `k3s/sky/ray-gpu-training-{provider}.yaml` |
| Submit LLM training | Triggers `llm_training_pipeline` DAG → uses `k3s/sky/ray-llm-training-{provider}.yaml` |
| Submit serving | Triggers `serving_pipeline` DAG → patches `k3s/kuberay/serving/rayservice-model-serving.yaml` |
| Submit vLLM | Triggers `vllm_serving_pipeline` DAG → uses `k3s/sky/ray-vllm-multinode-serving.yaml` |

Params are passed as `dag_run.conf` dict; the DAG reads keys set by `job_builder.py`.

## src/ ↔ k3s/

`src/dsl/` runs inside:
- `k3s/spark/spark-application.yaml` (Spark pod)
- `k3s/kuberay/kuberay-job.yaml` (Ray job pod, for online executor init)

`src/serve/` runs inside:
- `k3s/kuberay/serving/rayservice-model-serving.yaml` (RayService pod)

`src/pipeline/` runs inside:
- SkyPilot-launched cloud pod (referenced by `k3s/sky/ray-gpu-training-*.yaml` run command)

`src/services/` runs inside:
- FastAPI pod (`dsl-app`) — the backend imports these at runtime

## k3s/ DAG → SkyPilot YAML routing

Routing logic lives in `k3s/airflow/dags/sky_runner.py` and `app/backend/services/job_builder.py`:

| (kind, provider) | YAML |
|------------------|------|
| train, runpod | `ray-gpu-training-runpod.yaml` |
| train, vast | `ray-gpu-training-vast.yaml` |
| train_multi, aws | `ray-gpu-multinode-aws.yaml` |
| llm, runpod | `ray-llm-training-runpod.yaml` |
| llm, vast | `ray-llm-training-vast.yaml` |
| llm, aws | `ray-llm-training-aws.yaml` |
| vllm, runpod | `vllm-serving-runpod.yaml` |
| vllm, vast | `vllm-serving-vast.yaml` |
| vllm_multi, aws | `ray-vllm-multinode-serving.yaml` |

## app/ ↔ producer/

No direct coupling. Producer is a standalone Kafka producer for test data.
- Uses same Kafka topic (`topic-traffic`) that Spark Kafka connector reads from.
- Only dependency: KAFKA_BOOTSTRAP_SERVERS env var must point to the same cluster.

## Cross-layer Data Flow — Full Pipeline

```
1. Upload (S3)          → app/backend/routers/datasets.py
2. Ingest (Spark → Iceberg) → k3s/airflow/dags/* → k3s/spark/
3. Build DSL (S3)       → app/backend/routers/dsls.py
4. Preprocess (Spark)   → preprocessing_pipeline → src/dsl/ (in Spark pod)
5. Train (Ray/SkyPilot) → training_pipeline_skypilot → src/pipeline/ (in cloud pod)
6. Register (MLflow)    → src/pipeline/utils/mlflow_utils.py (from cloud pod)
7. Promote (MLflow)     → serving_pipeline → src/serve/registry.py
8. Serve (Ray Serve)    → rayservice-model-serving → src/serve/runtime.py
9. Infer (online)       → src/serve/runtime.py → src/dsl/numpy_executor.py
```

## Stale / Disconnected Wiring

- `checkArtifact()` in `platformClient.ts` → never called by any page
- `uploadSchemas()` (batch) in `platformClient.ts` → never called; individual `uploadFullSchema/Raw` are used instead
- `listProcessingRuns()` in `platformClient.ts` → defined but not called
- `k3s/airflow/dags/dag.py` → empty stub, no functionality
- `app/backend/routers/datasets.py:POST /{name}/schemas` → legacy batch endpoint, pages use `/api/v2/schemas/{dataset}/full|raw` instead
