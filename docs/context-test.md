# GPU Orchestration Platform — LLM Debugging Context

Feed this document to an LLM when something breaks and you want help diagnosing and fixing it.
It covers architecture, fixed bugs, S3 paths, DAG conf requirements, and common failure modes.

---

## Platform Overview

A Kubernetes-based MLOps platform that runs ML training workloads on external cloud VMs
(RunPod, Vast.ai, AWS) via SkyPilot, orchestrated by Airflow 3.x, with MLflow for experiment
tracking and Grafana/Prometheus for metrics.

**Key components:**
- `app/backend/` — FastAPI backend (Python 3.12)
- `app/frontend/` — React/TypeScript frontend
- `k3s/airflow/dags/` — Airflow 3.x DAGs
- `k3s/sky/*.yaml` — SkyPilot task YAML files (managed jobs for training, sky.launch for serving)
- `k3s/kuberay/main.py` — Training entrypoint (tabular: ANN, SSM, BAE, XGBoost, PyTorch)
- `k3s/kuberay/llm_main.py` — Training entrypoint (LLM fine-tuning)
- `src/pipeline/` — Ray Train framework code
- `src/services/gpu_catalog.py`, `src/services/gpu_selector.py` — GPU availability and scoring
- `app/backend/services/orchestration_selector.py` — Routes to correct DAG/YAML
- `app/backend/services/job_builder.py` — Builds Airflow dag_conf (no file writing)

---

## Critical Architecture Facts

### Airflow Version and Auth
**Airflow 3.x** — REST API uses `/api/v2/` (NOT `/api/v1/` which is Airflow 2.x).
Authentication: JWT Bearer token via `POST /auth/token` endpoint. Basic auth (`auth=(user, pass)`)
does NOT work in Airflow 3.x.

The shared helper `_airflow_request()` in `app/backend/routers/runs.py` handles both paths:
- In-cluster (inside K8s): direct HTTP to `airflow-api-server.airflow.svc.cluster.local`
- Out-of-cluster: Kubernetes API proxy mechanism

### MLflow Access
- **In-cluster** (from Airflow pod, Ray workers inside K8s): `http://my-mlflow.ray.svc.cluster.local:80`
- **External VMs** (RunPod/Vast/AWS): need a **public URL** — K8s internal DNS does NOT resolve outside the cluster
- All SkyPilot YAML defaults have `MLFLOW_TRACKING_URI: ""` — the Airflow DAG injects the correct value via `task.update_envs()` from the pod env var `MLFLOW_TRACKING_URI`

### S3 Path Conventions
```
runs/preprocessing/{pre_run_id}/params_preprocess.yaml
runs/training/{train_run_id}/params_training.yaml
runs/serving/{serve_run_id}/params_serving.yaml
runs/serving/{serve_run_id}/endpoint.json          ← vLLM endpoint URL written by DAG
mlflow-artifacts/                                   ← MLflow artifacts
warehouse/                                          ← Iceberg tables
```

### Run ID Formats
```
pre-{dataset}-{timestamp}Z-{hex6}                    ← preprocessing
train-{dataset}-{pre_suffix6}-{timestamp}Z-{hex6}    ← training
serve-{dataset}-{train_suffix6}-{timestamp}Z-{hex6}  ← serving
```

### DAG IDs
```
preprocessing_pipeline           ← Spark preprocessing (Kubernetes SparkApplication)
training_pipeline                ← in-cluster Ray (no GPU, uses KubeRay)
training_pipeline_skypilot       ← external GPU via SkyPilot managed jobs (sky.jobs.launch)
llm_training_pipeline            ← LLM fine-tuning via SkyPilot
serving_pipeline                 ← tabular model RayServe deploy + MLflow champion promotion
vllm_serving_pipeline            ← single-node vLLM (sky.launch, persistent cluster)
ray_vllm_serving_pipeline        ← multi-node Ray+vLLM (sky.launch, 2+ nodes)
```

---

## Required DAG Conf Keys

### training_pipeline_skypilot (CRITICAL: all these must be present)
```python
{
    "train_run_id": "train-network_traffic-abc123-20260323T120000Z-def456",
    "train_params_s3_path": "s3://k8s-mlops-platform-bucket/runs/training/{id}/params_training.yaml",
    "preprocess_run_id": "pre-network_traffic-...",  # can be empty string
    "processed_table": "iceberg.processed.network_traffic_abc123",
    "model_type": "ann",  # ann|ssm|bae|xgboost|pytorch
    "use_gpu": True,
    "num_nodes": 1,
    "resource_constraints": {  # optional; if present, DAG queries live GPU catalog
        "providers": ["runpod"],
        "prefer_spot": True,
        "min_vram_gb": 8,
        "num_gpus_per_node": 1,
    }
}
```
**The DAG does `conf["train_params_s3_path"]` at line 186 — KeyError if missing.**

### llm_training_pipeline
```python
{
    "llm_train_run_id": "train-llm-...",
    "llm_model_id": "Qwen/Qwen2.5-7B",
    "train_dataset_s3": "s3://bucket/datasets/llm/...",
    "max_steps": 100,
    "lora_enabled": True,
    # resource_constraints same as above
}
```

### vllm_serving_pipeline
```python
{
    "serve_run_id": "serve-llm-...",
    "llm_model_id": "Qwen/Qwen2.5-7B",
    "vllm_port": 8000,
    "max_model_len": 4096,
    "tensor_parallel_size": 1,
    "hf_token": "",  # empty for ungated models
}
```

---

## params_training.yaml Required Structure

The DAG reads this file from S3 at `train_params_s3_path`. If it's malformed or missing
blocks, training will fail silently or raise a KeyError deep in `main.py`.

```yaml
run_metadata:
  run_id: train-network_traffic-abc123-...
  run_type: training
  dataset: network_traffic

execution:
  train_run_id: train-network_traffic-abc123-...
  processed_table: iceberg.processed.network_traffic_abc123

lineage:
  preprocess_run_id: pre-network_traffic-...  # empty string if none
  processed_table: iceberg.processed.network_traffic_abc123
  dataset: network_traffic   # REQUIRED — serving_configs.py reads this to build serve_run_id

kuberay:
  model:
    framework: ann             # ann|ssm|bae|xgboost|pytorch
    use_gpu: true
    target: label
    num_classes: 6
    task_type: classification
    input_dim: 14
    mlflow_tracking_uri: "http://YOUR_PUBLIC_MLFLOW_URL"
    mlflow_experiment_name: "my_experiment"
    mlflow_artifact_location: "s3://k8s-mlops-platform-bucket/mlflow-artifacts/"
    mlflow_registry_model_name: "my_model"

hyperparams:
  ann: {}                     # empty = use defaults; or {"hidden_dim": 128, "lr": 0.001}

iceberg_tables:
  warehouse: "s3://k8s-mlops-platform-bucket/warehouse"
  processed:
    catalog: iceberg
    namespace: processed
```

---

## Known Bugs — All Fixed (do not re-introduce)

### Bug 1: MLflow K8s Internal DNS in SkyPilot YAMLs
**Files:** `k3s/sky/vast-test.yaml`, `k3s/sky/ray-gpu-training.yaml`,
`k3s/sky/ray-llm-training.yaml`, `k3s/sky/ray-gpu-multinode-aws.yaml`

**Problem:** Default had `MLFLOW_TRACKING_URI: "http://my-mlflow.ray.svc.cluster.local:80"`.
External VMs (RunPod/Vast/AWS) cannot resolve K8s cluster-internal DNS — they silently
run without logging any experiments.

**Fix:** All YAML defaults now `MLFLOW_TRACKING_URI: ""`. The Airflow DAG injects the
correct public URL via `task.update_envs()`.

**What you need:** `MLFLOW_TRACKING_URI` env var on the Airflow pod must be a publicly
reachable URL (not the K8s DNS).

**Symptom if misconfigured:** Training completes successfully but no run appears in MLflow.
The VM logs show `MLflow tracking URI: ""` or the K8s DNS address.

---

### Bug 2: Airflow API v1 Used in jobs.py
**File:** `app/backend/routers/jobs.py`

**Problem:** All four Airflow helper functions used `/api/v1/dags/{dag_id}/dagRuns`.
Airflow 3.x uses `/api/v2/`. Returns HTTP 404.

**Fix:** All Airflow calls replaced with `_call_airflow` imported from `runs.py`:
```python
from routers.runs import _airflow_request as _call_airflow
```
This function uses `/api/v2/` paths.

**Symptom if re-introduced:** 404 response from Airflow when launching jobs via the
Launch Wizard. Browser devtools shows 404 on the `/api/v1/dags/...` URL.

---

### Bug 3: Basic Auth Instead of JWT in jobs.py
**File:** `app/backend/routers/jobs.py`

**Problem:** Used `requests.post(..., auth=(user, password))`. Airflow 3.x requires
JWT Bearer token via `POST /auth/token` first.

**Fix:** Uses `_airflow_request` from `runs.py` which handles JWT auth internally.

**Symptom if re-introduced:** HTTP 401 Unauthorized from Airflow.

---

### Bug 4: Missing train_params_s3_path in dag_conf
**File:** `app/backend/routers/jobs.py` → `launch_job()`

**Problem:** `training_pipeline_skypilot` DAG does `conf["train_params_s3_path"]`
(hard-coded in the DAG). The original `jobs.py` never generated or uploaded this.

**Fix:** Added `_upload_training_params()` that:
1. Builds `params_training.yaml` content from request fields + optional preprocess lineage
2. Uploads to S3 at `runs/training/{train_run_id}/params_training.yaml`
3. Includes the S3 path in dag_conf as `train_params_s3_path`

**Symptom:** `KeyError: 'train_params_s3_path'` in Airflow task logs (first task of the DAG).

---

### Bug 5: job_builder.py Wrote Temp YAML to Wrong Pod
**File:** `app/backend/services/job_builder.py`

**Problem:** Wrote SkyPilot task YAML to `/tmp/sky_job_{run_id}.yaml` on the **backend pod**.
The Airflow DAG runs on the **Airflow pod** (different K8s pod/container) and never reads
this file. Also: `sky_yaml_path` was in dag_conf but the DAG doesn't use it — it has its
own YAML routing logic.

**Fix:** Removed all `yaml.dump()` file writes. Both `build_training_job()` and
`build_serving_job()` now return `("", dag_conf)`.

**Note:** `job_builder.py` is now a pure function — it builds the `dag_conf` dict only.
The Airflow DAG picks the correct SkyPilot YAML based on `model_type` and `num_nodes`.

---

### Bug 6: XGBoost Trains on CPU Even When GPU Requested
**File:** `src/pipeline/utils/xgboost_utils.py` → `train_func()`

**Problem:** XGBoost >= 2.0 (current version: 3.1.3) requires `device="cuda"` in the
params dict. Ray's `ScalingConfig(use_gpu=True)` allocates the GPU for the worker but
XGBoost itself ignores it without this param.

**Fix:** 3 lines added after `params["nthread"] = cpus_per_worker`:
```python
# GPU support — XGBoost >= 2.0 uses device="cuda" (tree_method="hist" stays the same)
if os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes"):
    params["device"] = "cuda"
```

**Symptom without fix:** `nvidia-smi dmon` shows 0% GPU utilization (sm%) during XGBoost
training, despite the GPU being allocated. Training time is the same as CPU mode.

---

### Bug 7: Missing lineage Block Breaks Serving Config Creation
**File:** `app/backend/routers/jobs.py` → `_upload_training_params()`

**Problem:** `serving_configs.py` `submit_serving_config()` reads
`training_params["lineage"]["dataset"]` to construct the `serve_run_id`. When the Launch
Wizard launches training without a `preprocess_run_id`, no lineage block was written to
`params_training.yaml` → 400 error on serving config creation.

**Fix:** `_upload_training_params()` always writes a `lineage` block. When no
`preprocess_run_id` is available:
```python
lineage = {
    "preprocess_run_id": "",
    "processed_table": processed_table,
    "dataset": dataset,  # critical: must be non-empty for serving config to work
}
```

**Symptom:** `400 Could not resolve dataset from training params lineage block` when
trying to create a serving config for a model trained via the Launch Wizard.

---

### Bug 8: No idle_minutes_to_autostop on Serving Clusters
**Files:** `k3s/sky/vllm-serving.yaml`, `k3s/sky/ray-vllm-multinode-serving.yaml`

**Problem:** `sky.launch()` creates a **persistent** cluster. If the operator forgets to
`sky down` after serving, the cluster runs at on-demand pricing indefinitely.

**Fix:** Added `idle_minutes_to_autostop: 60` to both serving YAMLs.

---

## Common Failure Modes and Diagnosis Steps

### "KeyError: 'train_params_s3_path'" in training_dag_skypilot.py
1. Check: did the trigger come from `jobs.py` (Launch Wizard) or `runs.py` (Run Pipeline)?
2. For Launch Wizard: check if `_upload_training_params()` ran (look for S3 object at
   `runs/training/{id}/params_training.yaml`):
   ```bash
   aws s3 ls s3://k8s-mlops-platform-bucket/runs/training/ --recursive | grep params
   ```
3. Check Airflow DAG run conf in the UI — `train_params_s3_path` key must be present.
4. If absent: the `_upload_training_params()` function threw an exception before the DAG
   trigger. Check the backend pod logs.

### Training Completes but No MLflow Run Appears
1. Check `MLFLOW_TRACKING_URI` on the Airflow pod:
   ```bash
   kubectl exec -n airflow deploy/my-airflow-api-server -- env | grep MLFLOW
   ```
2. If it's the K8s DNS (`my-mlflow.ray.svc.cluster.local`), external VMs cannot reach it.
3. Check VM logs for `ConnectionRefusedError` or DNS resolution errors to MLflow.
4. The URI must be something like `http://your-public-domain/` or an ngrok URL.

### Airflow Returns 404 When Triggering DAG
1. Check the API path in the request. It must be `/api/v2/dags/{dag_id}/dagRuns`.
2. Check that the DAG ID exists: Airflow UI → DAGs tab → search for the exact name.
3. DAG IDs are case-sensitive.

### Airflow Returns 401 Unauthorized
1. Check JWT auth flow — `_airflow_request()` in `runs.py` POSTs to `/auth/token` first.
2. Verify `AIRFLOW_USER` and `AIRFLOW_PASSWORD` env vars on the backend pod:
   ```bash
   kubectl exec -n <backend-namespace> deploy/<backend-deployment> -- env | grep AIRFLOW
   ```

### "400 Could not resolve dataset from training params lineage block"
1. The `params_training.yaml` is missing the `lineage.dataset` field.
2. Download and inspect the file:
   ```bash
   aws s3 cp s3://k8s-mlops-platform-bucket/runs/training/{train_run_id}/params_training.yaml -
   ```
3. The `lineage` key must exist with non-empty `dataset`.
4. Fix: re-trigger training — `_upload_training_params()` now always writes the lineage block.

### XGBoost GPU Util = 0% During Training
1. Check `xgboost_utils.py` `train_func()`: look for `params["device"] = "cuda"` block.
2. Verify `USE_GPU=true` is being passed to the VM (check in Airflow DAG env injection).
3. Verify XGBoost version:
   ```bash
   sky exec {cluster} -- python3 -c "import xgboost; print(xgboost.__version__)"
   # Must be >= 2.0 for device="cuda" to be valid (current: 3.1.3)
   ```
4. Look for `device` in XGBoost logs: XGBoost 3.x prints the device it's using at init.

### vLLM Endpoint Never Becomes Healthy
1. Check VM logs for CUDA OOM (model too large for GPU VRAM):
   ```bash
   sky logs {cluster_name}
   # Look for: CUDA out of memory, or "kill" message
   ```
2. Check model download (HuggingFace errors for gated models without token):
   ```bash
   sky logs {cluster_name} | grep -i "huggingface\|token\|gated\|403"
   ```
3. Verify the correct port is being polled (DAG uses `VLLM_PORT` env var).
4. For testing use `Qwen/Qwen2.5-0.5B` (smallest model, 1 GB VRAM) to isolate the issue.

### Preprocessing Run Not in Dropdown
1. Check the API directly:
   ```bash
   curl http://your-app/api/v2/processing-runs/ids
   ```
2. Expected S3 structure: `runs/preprocessing/{run_id}/params_preprocess.yaml`
3. Legacy path (`params/{run_id}/params_preprocess.yaml`): still readable by fallback code.
4. If S3 is empty: preprocessing DAG may have failed before writing params — check Airflow.

### SkyPilot Job Hangs > 15 Minutes Provisioning
1. Check spot availability: `sky show-gpus --cloud runpod`
2. Check DAG is not waiting for a GPU SKU that has 0 availability.
3. The timeout guard in `_poll_sky_job()` should eventually cancel — but verify it's active
   (look for `SKY_TIMEOUT_SECONDS` usage in `training_dag_skypilot.py`).
4. Cancel manually: `sky jobs cancel {job_name} --yes`

---

## Key File Locations

| Component | File |
|-----------|------|
| Launch Wizard API | `app/backend/routers/jobs.py` |
| Run Pipeline API | `app/backend/routers/runs.py` |
| Serving configs API | `app/backend/routers/serving_configs.py` |
| Model architectures API | `app/backend/routers/model_architectures.py` |
| GPU catalog | `src/services/gpu_catalog.py` |
| GPU selector | `src/services/gpu_selector.py` |
| Orchestration selector | `app/backend/services/orchestration_selector.py` |
| Job builder | `app/backend/services/job_builder.py` |
| Tabular training entrypoint | `k3s/kuberay/main.py` |
| LLM training entrypoint | `k3s/kuberay/llm_main.py` |
| XGBoost worker (GPU fix here) | `src/pipeline/utils/xgboost_utils.py` |
| PyTorch worker + Prometheus | `src/pipeline/utils/pytorch_utils.py` |
| Tabular training DAG (SkyPilot) | `k3s/airflow/dags/training_dag_skypilot.py` |
| LLM training DAG | `k3s/airflow/dags/llm_training_dag.py` |
| vLLM serving DAG | `k3s/airflow/dags/vllm_serving_dag.py` |
| Multi-node vLLM DAG | `k3s/airflow/dags/ray_vllm_serving_dag.py` |
| Tabular serving DAG | `k3s/airflow/dags/serving_dag.py` |
| Airflow K8s helpers | `k3s/airflow/k8s_helpers.py` |
| Single-node vLLM YAML | `k3s/sky/vllm-serving.yaml` |
| Multi-node vLLM YAML | `k3s/sky/ray-vllm-multinode-serving.yaml` |
| Single-node GPU training YAML | `k3s/sky/ray-gpu-training.yaml` |
| Multi-node training YAML | `k3s/sky/ray-gpu-multinode-aws.yaml` |
| LLM training YAML | `k3s/sky/ray-llm-training.yaml` |
| Unit tests (routers) | `tests/test_routers.py` |
| Unit tests (model upload) | `tests/test_model_upload.py` |

---

## _airflow_request() Pattern (runs.py)

Both `runs.py` and `serving_configs.py` have this helper. `jobs.py` imports it from `runs.py`.

```python
# Simplified pattern:
def _airflow_request(method: str, path: str, body: dict = None):
    # Step 1: get JWT token
    token_resp = requests.post(
        f"{AIRFLOW_BASE_URL}/auth/token",
        json={"username": AIRFLOW_USER, "password": AIRFLOW_PASSWORD}
    )
    token = token_resp.json()["access_token"]

    # Step 2: make the actual call
    resp = requests.request(
        method,
        f"{AIRFLOW_BASE_URL}/{path}",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
    )
    return resp.status_code, resp.json()
```

In-cluster: `AIRFLOW_BASE_URL = http://my-airflow-api-server.airflow.svc.cluster.local:8080`
Out-of-cluster: uses Kubernetes API proxy

---

## SkyPilot Training Flow (Tabular)

```
App (jobs.py or runs.py)
  → fetch preprocess params from S3 (if preprocess_run_id given)
  → build lineage: processed_table, dataset, preprocess_run_id
  → generate params_training.yaml
  → upload to s3://bucket/runs/training/{train_run_id}/params_training.yaml
  → trigger training_pipeline_skypilot DAG with dag_conf including train_params_s3_path

  Airflow DAG (training_dag_skypilot.py):
    → _submit_sky_job():
        reads resource_constraints from conf
        queries GPUSelectorService for best available GPU
        selects SkyPilot YAML (ray-gpu-training.yaml or ray-gpu-multinode-aws.yaml)
        calls sky.jobs.launch(task, name=job_name, retry_until_up=True, detach_run=True)
    → _poll_sky_job():
        polls sky.jobs.queue() every 30s until SUCCEEDED or FAILED

  External VM (RunPod/Vast/AWS):
    setup: pip install Ray + dependencies
    run:
      ray start --head --num-gpus=N
      export PARAMS_S3_PATH=s3://bucket/runs/training/{id}/params_training.yaml
      python3 k3s/kuberay/main.py
        → downloads params_training.yaml from PARAMS_S3_PATH
        → connects to Iceberg/Glue to read processed table
        → Ray Train with ScalingConfig(num_workers=N, use_gpu=True)
        → logs to MLflow (needs MLFLOW_TRACKING_URI to be a public URL)
        → uploads model artifact to S3
```

---

## Environment Variables Required on Airflow Pod

```bash
MLFLOW_TRACKING_URI=http://your-public-mlflow-url   # CRITICAL — not K8s DNS
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_DEFAULT_REGION=us-east-2
S3_BUCKET=k8s-mlops-platform-bucket
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin
# SkyPilot config at ~/.sky/config.yaml (RunPod API key, Vast API key, AWS profile)
```

---

## SkyPilot YAML Design Principles

- **Managed jobs** (`sky.jobs.launch`): for training — spot-first, auto-retry on preemption
- **Persistent cluster** (`sky.launch`): for serving — on-demand only, `idle_minutes_to_autostop: 60`
- `MLFLOW_TRACKING_URI: ""` in all YAMLs — always injected by DAG, never hardcoded
- `any_of` resource list: defines fallback priority; first available spot is chosen
- `setup` block runs once per cluster; `run` block runs on every launch

---

## Model Architecture Validation (model_architectures.py)

The `_validate_architecture()` function uses Python's `ast` module (no code execution):
1. Parses the uploaded `.py` file for syntax errors → 422 if invalid
2. Finds all `class` definitions that inherit from `nn.Module` or `torch.nn.Module`
3. Checks each such class for a `forward` method → 422 if missing
4. If no class inherits from `nn.Module` at all → 422

Valid pattern:
```python
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, input_dim: int = 14, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)
```

---

## Unit Tests

All unit tests are hermetic (no network, no S3, no Airflow):

```bash
pytest tests/test_model_upload.py -v    # 7 tests: AST validation
pytest tests/test_routers.py -v         # 20 tests: orchestration routing, job builder, dry_run endpoint
# Total: 27 tests — all should pass in < 5 seconds
```

Test file locations: `tests/test_routers.py`, `tests/test_model_upload.py`
