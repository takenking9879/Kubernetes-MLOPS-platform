# app/back/ — Key Elements

## Routers

### datasets.py (`/api/v2/datasets`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | List datasets (S3 raw/ prefixes) |
| POST | `/` | Create dataset (write `.keep` to S3) |
| POST | `/{name}/upload` | Upload parquet files to S3 |
| POST | `/{name}/ingest` | Create SparkApplication CRD (K8s ingestion) |
| GET | `/{name}/ingest/{job}/status` | Poll SparkApplication state |
| GET | `/{name}/iceberg-schema` | Return columns from `iceberg.raw.{name}` |
| GET | `/{name}/sample` | Fetch up to 3000 rows from Iceberg |
| GET | `/from-iceberg` | Load schema from any Iceberg table |
| POST | `/{name}/schemas` | Upload raw/full/preprocessed YAML to S3 (legacy batch) |

Side effects: S3 writes, K8s API calls, pyiceberg queries

---

### dsls.py (`/api/v2/datasets/{name}/dsls`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/{name}/dsls` | List DSL versions in S3 |
| GET | `/{name}/dsls/{version}` | Fetch DSL YAML content |
| POST | `/{name}/dsls` | Save new DSL version (auto-increment, validates slug uniqueness) |

S3 path pattern: `dsl/dsl_{dataset}/v{N}__{slug}.yaml`

---

### processing_runs.py (`/api/v2/processing-runs`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/` | Trigger `preprocessing_pipeline` DAG; writes `params_preprocess.yaml` to S3 |
| GET | `/` | List processed tables from Iceberg metadata |
| GET | `/ids` | List preprocessing run IDs from S3 |
| GET | `/{run_id}/params` | Fetch `params_preprocess.yaml` from S3 |
| GET | `/{dag_run_id}/status` | Poll Airflow DAG state |

S3 path: `runs/preprocessing/{preprocess_run_id}/params_preprocess.yaml`

---

### runs.py (`/api/v2/runs`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/ids` | List training run IDs from S3 |
| GET | `/check` | Check if artifact_set_id exists in Iceberg |
| POST | `/` | Generate params_training.yaml, upload to S3, trigger Airflow DAG |
| GET | `/{dag_run_id}/status` | Poll Airflow DAG state |

DAGs triggered: `training_pipeline` (local) or `training_pipeline_skypilot` (cloud, `?skypilot=true`)
S3 path: `runs/training/{train_run_id}/params_training.yaml`

---

### schemas.py (`/api/v2/schemas`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/{dataset}/full` | Upload full.yaml with auto-versioning |
| POST | `/{dataset}/raw` | Upload raw.yaml with auto-versioning |

S3 path: `schemas/datasets/{dataset}/v{N}/{type}.yaml`

---

### serving_configs.py (`/api/v2/serving-configs`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/` | Generate params_serving.yaml, save to S3 |
| GET | `/{serve_run_id}/params` | Fetch params_serving.yaml from S3 |
| POST | `/{serve_run_id}/deploy` | Trigger `serving_pipeline` DAG |
| GET | `/{serve_run_id}/deploy/{dag_run_id}/status` | Poll DAG state |
| POST | `/vllm-deploy` | Trigger `vllm_serving_pipeline` DAG |
| GET | `/{serve_run_id}/endpoint` | Fetch registered vLLM endpoint URL from S3 |

---

### config.py (`/api/v2/config`)
Static manifest returning training capabilities:
- Frameworks: XGBoost, PyTorch
- Task types: classification, regression
- Loss functions, evaluation metrics (accuracy, F1, MSE, etc.)

---

### gpu_resources.py (`/api/v2/gpu-resources`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/catalog` | Unified GPU offers (spot-first); optional provider/min_vram filter |
| POST | `/select` | Return ranked SkyPilot `any_of` list from `ResourceConstraints` |
| GET | `/llm-catalog` | Static LLM model → GPU recommendations |

Module-level singletons: `_catalog_svc = GPUCatalogService()`, `_selector_svc = GPUSelectorService()`
`_offer_to_dict()`: serializes `GPUOffer` including `skypilot_supported` field.

LLM catalog static entries: Llama-3.1-8B, Qwen2.5-7B, Llama-3.1-70B, DeepSeek-R1-8B

---

### jobs.py (`/api/v2/jobs`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/launch` | Unified job launch; routes to orchestration_selector + job_builder; triggers DAG |
| GET | `/{job_id}/status` | Poll Airflow state |
| GET | `/` | List recent DAG runs |
| DELETE | `/{job_id}` | Cancel/mark job failed |

Orchestration types: `ray_train`, `llm_finetune`, `vllm_single_node`, `ray_vllm_multinode`

---

### model_architectures.py (`/api/v2/model-architectures`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | List built-in + user-uploaded architectures |
| POST | `/upload` | AST-validate + upload `.py` file; must define class inheriting `nn.Module` with `forward()` |

Built-in: MLP (ANN), SSM Tabular, BAE Ensemble, XGBoost
S3 storage: `model-architectures/{arch_id}/{upload_id}/{filename}`

---

## main.py
Function/Class: FastAPI app entrypoint
File: `app/backend/main.py`

Does:
- Mounts all 10 routers under `/api/v2/`
- Legacy endpoints: `POST /api/schema/from-csv`, `POST /api/schema/upload-file`, `GET /api/health`, `POST /api/dry-run`, `POST /api/validate-yaml`
- CORS: wide open (`*`)
- No authentication middleware

---

## services/orchestration_selector.py
Function/Class: `generate_recommendations(job_type, model_type, model_vram_gb, num_nodes, num_gpus_per_node, vram_per_gpu_gb)`
File: `app/backend/services/orchestration_selector.py`

Does:
- Routes to `llm_finetune` if model_type == "llm"
- Routes to `vllm_single_node` or `ray_vllm_multinode` for serving based on VRAM fit
- Returns `OrchestratorRecommendation` (orchestration_type, dag_id, sky_yaml_label, reasoning)

DAG IDs:
- `training_pipeline_skypilot` (tabular)
- `llm_training_pipeline` (LLM)
- `vllm_serving_pipeline` (vLLM single node)
- `ray_vllm_serving_pipeline` (multi-node)

---

## services/job_builder.py
Function/Class: `JobBuilder`
File: `app/backend/services/job_builder.py`

Does:
- `build_training_job(config: TrainingJobConfig)` → `(yaml_path, dag_conf)`
  - Loads provider-specific base YAML from `k3s/sky/`
  - Injects `any_of` list from GPUSelectorService
  - Returns Airflow `dag_run.conf` dict
- `build_serving_job(config: ServingJobConfig, multinode)` → `(yaml_path, dag_conf)`
- `yaml_preview(config)` → str (for UI preview)

YAML routing table: maps `(kind, provider)` → filename in `k3s/sky/`
