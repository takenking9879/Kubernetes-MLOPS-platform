# Architecture Overview

## High-Level Flow

```
User (React UI)
  ↓ REST calls (platformClient.ts)
FastAPI Backend (app/backend/)
  ├── S3 (write params_*.yaml, DSL YAMLs, schemas)
  ├── Iceberg (query metadata, processed tables)
  └── Airflow REST API (trigger DAG runs)
        ↓ dag_run.conf payload
  Airflow DAG (k3s/airflow/dags/)
        ├── SparkApplication (K8s CRD) → preprocessing
        ├── RayJob (K8s CRD, KubeRay) → local training
        └── KubernetesPodOperator → sky-runner pod → SkyPilot
              ↓ sky.jobs.launch() / sky.launch()
        Cloud GPU (RunPod / Vast.ai / AWS)
              ↓ Ray Train / TRL / vLLM
        Trained model / serving endpoint
              ↓
        MLflow Registry (S3 artifacts + metadata)
              ↓
        Ray Serve (RayService CRD, KubeRay)
              ↓ HTTP
        Inference responses
```

## Training Flow — Tabular (Classical ML)

1. User uploads parquets → `POST /api/v2/datasets/{name}/upload`
2. Ingest to Iceberg → `POST /api/v2/datasets/{name}/ingest` → SparkApplication (ingestion)
3. Build DSL YAML in visual editor → `POST /api/v2/datasets/{name}/dsls`
4. Submit preprocessing → `POST /api/v2/processing-runs` → triggers `preprocessing_pipeline` DAG
   - DAG: SparkApplication with dsl_*.yaml → writes to `iceberg.processed.{dataset}_{execId}`
5. Submit training → `POST /api/v2/runs` → triggers `training_pipeline_skypilot` DAG
   - Generates `params_training.yaml` → uploads to S3
   - DAG: sky-runner pod → SkyPilot → cloud GPU → Ray Train → MLflow
6. Submit serving → `POST /api/v2/serving-configs/{id}/deploy` → triggers `serving_pipeline` DAG
   - DAG: MLflow promote → patch RayService → (optional) Spark Kafka connector

## Training Flow — LLM

1. User selects LLM model + config in LaunchWizardPage
2. `POST /api/v2/jobs/launch` → `job_builder.py` generates `params_training.yaml`
3. Triggers `llm_training_pipeline` DAG
   - DAG: sky-runner → SkyPilot → cloud A100/H100 → TRL SFTTrainer + LoRA + DeepSpeed ZeRO

## Serving Flow — Ray Serve (Classical ML)

1. `serving_pipeline` DAG runs after training
2. MLflow: model tagged with `@champion` alias
3. RayService patched: `PARAMS_SERVING_S3_PATH` injected into runtime_env
4. Ray Serve: `k3s/kuberay/serving/rayservice-model-serving.yaml` uses `src/serve/`
5. Online inference: `ModelRuntime.predict()` handles pre-processed or raw payloads

## Serving Flow — vLLM

1. User submits vLLM deploy → `POST /api/v2/serving-configs/vllm-deploy`
2. Triggers `vllm_serving_pipeline` DAG
3. DAG: sky-runner → `sky.launch(detach_run=True)` → cloud GPU cluster
4. Endpoint URL written to S3 → polled by `GET /api/v2/serving-configs/{id}/endpoint`

## Online Inference Paths (Ray Serve)

```
Kafka Event (JSON)  →  Ray Serve endpoint
  Payload {"raw": {...}}
    → NumpyPipelineExecutor.transform_to_vector()
    → XGBoostAdapter / PyTorchAdapter
    → Predictions

  Payload {"data": [[f1...f14]]}  (pre-processed)
    → direct to adapter (no DSL)
```

## Canary Deployment

```
MLflow alias.created/deleted webhook
  → Ray Serve handler
  → Reload ModelRuntime (new alias)
  → TrafficRouter.set_canary_probability()
  → route: stable (1-p) vs canary (p)
```

## GPU Selection (gpu_catalog + gpu_selector)

```
User constraints (ResourceConstraints)
  → GPUCatalogService.query_availability()
      ├── RunPod GraphQL API (real-time)
      ├── Vast.ai REST API (real-time)
      └── SkyPilot catalog + boto3 spot prices (AWS)
  → GPUSelectorService.select_providers()
      ├── Hard filter: skypilot_supported=True only
      ├── Score: spot * bonus < on_demand * 2.5 * bonus
      └── Build infra paths per provider
  → SkyPilot any_of list → injected into job YAML
```

## DSL Pipeline — Dual Execution

| Context | Executor | When |
|---------|----------|------|
| Training/preprocessing | Spark (`Pipeline.fit_transform()`) | Batch, offline |
| Online inference (Ray Serve) | `NumpyPipelineExecutor` | Real-time, per-request |

Same DSL YAML drives both paths. NumpyExecutor mirrors all Spark transformers in pure Python.

## Params / Config Hierarchy

```
S3: runs/preprocessing/{id}/params_preprocess.yaml   → preprocessing run
S3: runs/training/{id}/params_training.yaml          → training run
S3: runs/serving/{id}/params_serving.yaml            → serving run
S3: schemas/datasets/{dataset}/v{N}/full.yaml        → feature schema (versioned)
S3: dsl/dsl_{dataset}/v{N}__{slug}.yaml              → DSL version (versioned)
```

Legacy: `k3s/params.yaml` — developer reference only, not used at runtime.

## Deployment Topology

| Component | Platform |
|-----------|----------|
| FastAPI backend | K8s pod (`dsl-app`) |
| React frontend | Served by backend (or separate nginx) |
| Airflow | Helm release in K8s (`k3s/airflow/`) |
| SparkApplication | K8s CRD via Spark Operator |
| RayJob / RayService | K8s CRDs via KubeRay operator |
| SkyPilot jobs | Cloud (RunPod / Vast.ai / AWS EC2) |
| MLflow server | K8s pod (`k3s/mlflow/`) |
| Prometheus / Grafana | K8s (`k3s/`) |
