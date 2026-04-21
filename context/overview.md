# MLOps Platform — Overview

## What This Is
A SaaS MLOps orchestrator for multi-cloud GPU compute. Users upload datasets, build reusable Spark-based feature engineering pipelines, train classical ML or LLM models, and deploy them for serving — all from a React UI. The platform handles infra abstraction (SkyPilot, Ray, KubeRay, Spark-on-K8s) and shows real-time GPU pricing across RunPod, Vast.ai, and AWS.

## Key Layers

| Layer | Path | Role |
|-------|------|------|
| Frontend | `app/frontend/` | React SPA — ZENTHROSML CANVAS DAG UI, 2 pages, Zustand state, REST calls to backend |
| Backend | `app/backend/` | FastAPI — 10 routers, triggers Airflow + writes to S3/Iceberg |
| Core logic | `src/` | DSL pipelines, serving runtime, GPU catalog, training pipeline |
| Orchestration | `k3s/` | Airflow DAGs, SkyPilot YAMLs, Spark/KubeRay manifests |
| Data producer | `producer/` | Synthetic Kafka traffic generator for testing |

## Frontend Pages (post-ZENTHROSML CANVAS redesign)

| Page | Route | Purpose |
|------|-------|---------|
| OrchestrationCanvasPage | `canvas` (default) | Visual DAG pipeline editor — node-based orchestration with right-panel inspector |
| DSL Builder (MainLayout) | `dsl-builder` | Feature engineering pipeline editor (ReactFlow + pipelineStore) |

The canvas has 4 active node types: Dataset, Processing, Training, Serving.  
Legacy pages (DatasetPage, ProcessingPage, LaunchWizardPage) are removed from top-nav; their logic is embedded in node inspectors.  
See `context/dag_multi_input_future.md` for planned future nodes.

## Frontend State Stores

| Store | File | Scope |
|-------|------|-------|
| `uiStore` | `src/store/uiStore.ts` | Active page (`canvas` or `dsl-builder`) |
| `dagStore` | `src/store/dagStore.ts` | Orchestration DAG nodes, edges, artifact propagation, run execution |
| `pipelineStore` | `src/store/pipelineStore.ts` | DSL Builder node graph, validation, dry-run |
| `datasetStore` | `src/store/datasetStore.ts` | Active dataset for DSL Builder |

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
| tabular_serving_skypilot_pipeline | API | SkyPilot sky serve tabular model — single/multi-node Ray Serve |
| serving_pipeline | API | MLflow promote → patch RayService → optional Kafka (in-cluster) |
| model_promotion_workflow | Manual/auto | MLflow alias management |

## Data Stores

| Store | Contents |
|-------|----------|
| S3 | Raw parquets, DSL YAMLs, params_*.yaml, model architectures, schemas; trained models at `v1/models/{registry_name}/{train_run_id}/model_{type}.pkl` + `model_metadata.json` sidecar |
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
