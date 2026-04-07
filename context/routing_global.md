# Global Routing Guide

Use this file to navigate to the right code area. Read overview.md first.

## By User Intent

### "Add / change a UI page or component"
→ `app/frontend/src/pages/` (pages) or `app/frontend/src/components/` (shared)
→ `app/frontend/src/api/platformClient.ts` (add/change API call)
→ `app/frontend/src/store/` (state, if new global state needed)
→ See: `context/app/front/`

### "Add / change a backend API endpoint"
→ `app/backend/routers/` (find or create router file for the domain)
→ `app/backend/main.py` (mount new router if new file)
→ See: `context/app/back/`

### "Change feature engineering (DSL transforms)"
→ `src/dsl/transformers.py` (stateless transforms)
→ `src/dsl/estimators.py` (learned transforms like scalers/encoders)
→ `src/dsl/numpy_executor.py` (mirror change here for online inference)
→ `src/dsl/state_registry.py` (register new type)
→ See: `context/src/dsl/`

### "Change online inference / prediction logic"
→ `src/serve/runtime.py` (`ModelRuntime.predict()`, `_build_matrix()`)
→ `src/serve/adapters.py` (XGBoost or PyTorch inference)
→ `src/serve/pipeline_loader.py` (DSL artifact loading chain)
→ See: `context/src/serve/`

### "Add a GPU provider or fix GPU pricing"
→ `src/services/gpu_catalog.py` (`_query_{provider}()` method)
→ `src/services/_sky_catalog_query.py` (SkyPilot catalog subprocess)
→ `src/services/gpu_selector.py` (scoring/ranking logic)
→ `app/backend/routers/gpu_resources.py` (serialisation)
→ See: `context/src/services/`

### "Add / modify an Airflow DAG"
→ `k3s/airflow/dags/` (add new DAG file)
→ `k3s/airflow/k8s_helpers.py` (K8s helpers, idempotent delete)
→ See: `context/k3s/`

### "Add or change a SkyPilot job YAML"
→ `k3s/sky/` (naming: `ray-{purpose}-{provider}.yaml`)
→ `app/backend/services/job_builder.py` (routing table maps (kind, provider) → YAML)
→ `k3s/airflow/dags/sky_runner.py` (dynamic YAML selection)
→ See: `context/k3s/`

### "Change model training (Ray Train)"
→ `src/pipeline/base_trainer.py` (abstract orchestrator)
→ `src/pipeline/train/pytorch.py` or `xgboost.py`
→ See: `context/src/`

### "Change serving config / deployment flow"
→ `app/backend/routers/serving_configs.py` (API — handles both in-cluster and SkyPilot paths)
→ `k3s/airflow/dags/serving_dag.py` (in-cluster: MLflow promote → patch RayService)
→ `k3s/kuberay/serving/rayservice-model-serving.yaml` (K8s RayService manifest — in-cluster)
→ `src/serve/config.py` (ServingConfig loading)
→ See: `context/k3s/`

### "Change tabular SkyPilot out-cluster serving"
→ `app/backend/routers/serving_configs.py` — `deployment_target='skypilot'` branch in `deploy_serving_config`
→ `k3s/airflow/dags/tabular_serving_skypilot_dag.py` — DAG: launch-tabular-serve → wait-for-endpoint → register-endpoint
→ `k3s/sky/tabular-serving-single.yaml` — sky serve YAML (single-node replica, Ray Serve)
→ `k3s/sky/tabular-serving-multinode.yaml` — sky serve YAML (multi-node replica, Kimi-K2 Ray pattern)
→ `k3s/sky/sky_runner.py` — `launch_tabular_serve()`, `wait_tabular_serve()` (uses `sky.serve.up()` + `sky.serve.status()`)
→ Frontend: `LaunchWizardPage.tsx` — tabServDeployTarget toggle, replica policy fields, endpoint polling

### "Add a new model architecture (custom PyTorch)"
→ `app/backend/routers/model_architectures.py` (upload + AST validation)
→ `app/frontend/src/pages/LaunchWizardPage.tsx` (upload UI, currently not wired)
→ See: `context/app/back/missing_features.md`

### "Change schema versioning"
→ `app/backend/routers/schemas.py` (`POST /api/v2/schemas/{dataset}/full|raw`)
→ `app/backend/routers/datasets.py` (`POST /api/v2/datasets/{name}/schemas` — legacy batch)

### "Add Kafka / streaming inference"
→ `k3s/spark/inference/spark-kafka-application.yaml` (Spark Kafka connector manifest)
→ `k3s/airflow/dags/serving_dag.py` (kafka branch logic)
→ `src/converters/spark_kafka_helper.py` (Spark schema converter)
→ `src/converters/raw_to_features.py` (Python online converter)

### "Change MLflow model promotion"
→ `k3s/airflow/dags/serving_dag.py` (`promote_to_champion` task)
→ `k3s/airflow/dags/promotion_dag.py` (manual / auto promotion)
→ `src/serve/registry.py` (`MLflowRegistry.load_by_alias()`)

## By File Type

| File type | Location |
|-----------|----------|
| React pages | `app/frontend/src/pages/*.tsx` |
| React components | `app/frontend/src/components/` |
| API client functions | `app/frontend/src/api/platformClient.ts` |
| TypeScript types | `app/frontend/src/api/platformClient.ts` (inline interfaces) |
| Zustand stores | `app/frontend/src/store/` |
| FastAPI routers | `app/backend/routers/*.py` |
| Backend services | `app/backend/services/*.py` |
| Airflow DAGs | `k3s/airflow/dags/*.py` |
| SkyPilot YAMLs | `k3s/sky/*.yaml` |
| Spark manifests | `k3s/spark/*.yaml` |
| KubeRay manifests | `k3s/kuberay/*.yaml` |
| DSL logic | `src/dsl/*.py` |
| Serving logic | `src/serve/*.py` |
| GPU catalog/selector | `src/services/gpu_*.py` |
| Training logic | `src/pipeline/` |
| Data converters | `src/converters/` |
| Synthetic data | `producer/` |

## Cross-Layer Consistency Rules

When you change one of these, you MUST also update:

| You change | Also update |
|-----------|-------------|
| `GPUOffer` dataclass fields | `gpu_resources.py:_offer_to_dict()` + `platformClient.ts:GPUOffer` |
| Backend endpoint schema | `platformClient.ts` (request/response types) |
| Airflow DAG conf keys | Backend router that triggers it (they set the conf dict) |
| DSL transformer (Spark) | `numpy_executor.py` (mirror for online inference) |
| SkyPilot YAML filename | `job_builder.py` routing table |
| New Airflow DAG | Backend router that triggers it + update DAG ID constant |
| `params_training.yaml` structure | `k3s/airflow/dags/training_dag_skypilot.py` (reads conf keys) |
| `params_serving.yaml` structure | `k3s/airflow/dags/serving_dag.py` + `src/serve/config.py` |
| Frontend page type | `app/frontend/src/store/uiStore.ts` + `App.tsx` |
