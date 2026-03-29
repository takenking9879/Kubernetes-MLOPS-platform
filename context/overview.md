# MLOps Platform — Overview

## What This Is
A SaaS MLOps orchestrator for multi-cloud GPU compute. Users upload datasets, build reusable Spark-based feature engineering pipelines, train classical ML or LLM models, and deploy them for serving — all from a React UI. The platform handles infra abstraction (SkyPilot, Ray, KubeRay, Spark-on-K8s) and shows real-time GPU pricing across RunPod, Vast.ai, and AWS.

## Key Layers

| Layer | Path | Role |
|-------|------|------|
| Frontend | `app/frontend/` | React SPA — 6 pages, Zustand state, REST calls to backend |
| Backend | `app/backend/` | FastAPI — 10 routers, triggers Airflow + writes to S3/Iceberg |
| Core logic | `src/` | DSL pipelines, serving runtime, GPU catalog, training pipeline |
| Orchestration | `k3s/` | Airflow DAGs, SkyPilot YAMLs, Spark/KubeRay manifests |
| Data producer | `producer/` | Synthetic Kafka traffic generator for testing |

## 6 Frontend Pages

| Page | Tab | Purpose |
|------|-----|---------|
| DatasetPage | datasets | Upload parquets → ingest to Iceberg |
| DSL Builder | dsl-builder | Visual pipeline editor (ReactFlow) |
| ProcessingPage | processing | Run Spark preprocessing DSL |
| RunPage | run-pipeline | Train model (XGBoost / PyTorch) |
| ServingPage | serving | Deploy to Ray Serve or vLLM |
| LaunchWizardPage | launch | Unified GPU job wizard (tabular / LLM) |

## 10 Backend Routers

| Prefix | Purpose |
|--------|---------|
| `/api/v2/datasets` | Dataset CRUD, upload, Iceberg ingest |
| `/api/v2/datasets/{name}/dsls` | Versioned DSL YAML management |
| `/api/v2/processing-runs` | Spark preprocessing runs |
| `/api/v2/runs` | Training runs |
| `/api/v2/schemas` | Versioned schema uploads |
| `/api/v2/serving-configs` | Serving configs + deploy |
| `/api/v2/config` | Static training capabilities manifest |
| `/api/v2/gpu-resources` | Live GPU catalog + selection |
| `/api/v2/jobs` | Unified job launch (Phase 8) |
| `/api/v2/model-architectures` | Custom PyTorch model upload |

## Orchestration (k3s/airflow/dags/)

| DAG | Trigger | What it does |
|-----|---------|--------------|
| preprocessing_pipeline | API | Spark DSL fit+transform → Iceberg |
| training_pipeline | API | KubeRay local training |
| training_pipeline_skypilot | API | SkyPilot GPU training (spot-first) |
| llm_training_pipeline | API | SkyPilot LLM fine-tuning (TRL+LoRA+DeepSpeed) |
| full_ml_pipeline | API | Spark → KubeRay (sequential) |
| ml_pipeline | API/UI | Flexible mode (preprocess only / train only / both) |
| vllm_serving_pipeline | API | SkyPilot multi-node Ray+vLLM serving |
| serving_pipeline | API | MLflow promote → patch RayService → optional Kafka |
| model_promotion_workflow | Manual/auto | MLflow alias management |

## Data Stores

| Store | Contents |
|-------|----------|
| S3 | Raw parquets, DSL YAMLs, params_*.yaml, model architectures, schemas |
| Apache Iceberg | Raw/processed feature tables, preprocessing_artifacts metadata |
| MLflow Registry | Trained models + aliases (champion/challenger) |
| K8s / Airflow | Orchestration state, SparkApplication/RayJob CRDs |

## Key src/ Modules

| Module | Purpose |
|--------|---------|
| `src/dsl/` | Spark DSL transformers + estimators + pipeline orchestration |
| `src/dsl/numpy_executor.py` | Pure-Python online DSL execution (Ray Serve, no Spark) |
| `src/serve/` | ModelRuntime, canary router, MLflow model loading |
| `src/converters/` | Raw Kafka events → feature dicts (Spark + Python) |
| `src/services/gpu_catalog.py` | Real-time GPU offers (RunPod + Vast + AWS) |
| `src/services/gpu_selector.py` | Spot-first SkyPilot `any_of` ranking |
| `src/pipeline/` | Ray Train distributed training (PyTorch + XGBoost) |

## Quick Navigation

- Modify an API endpoint → `app/backend/routers/`
- Add a frontend page or component → `app/frontend/src/pages/` or `components/`
- Change feature engineering logic → `src/dsl/`
- Change serving/inference logic → `src/serve/`
- Add a GPU provider or fix pricing → `src/services/gpu_catalog.py`
- Add or modify an Airflow DAG → `k3s/airflow/dags/`
- Change SkyPilot YAML (GPU job) → `k3s/sky/`
- Change K8s manifests (Spark/Ray) → `k3s/spark/` or `k3s/kuberay/`
