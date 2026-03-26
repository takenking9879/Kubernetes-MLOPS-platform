# End-to-End Test Guide — GPU Orchestration Platform

## Prerequisites (do these before ANY cloud test)

### 1. Run unit tests locally

```bash
cd /home/jorge/DocumentsWLS/Data_Science_Projects/Kubernetes-MLOPS-platform
pytest tests/test_model_upload.py tests/test_routers.py -v
# Expected: 27 passed
```

### 2. Set MLFLOW_TRACKING_URI on the Airflow pod

This is the single most critical prerequisite. Without it, every SkyPilot job on an
external VM will silently skip MLflow logging (K8s internal DNS is not reachable from
RunPod/Vast/AWS).

```bash
# Option A: patch the Airflow deployment env var
kubectl set env deployment/my-airflow-api-server -n airflow \
  MLFLOW_TRACKING_URI=http://YOUR_PUBLIC_MLFLOW_URL

# Option B: set as Airflow Variable
# Airflow UI → Admin → Variables → +
# Key: MLFLOW_TRACKING_URI   Value: http://YOUR_PUBLIC_MLFLOW_URL

# Verify it's set:
kubectl exec -n airflow deploy/my-airflow-api-server -- \
  python3 -c "import os; print(os.getenv('MLFLOW_TRACKING_URI'))"
# Expected: http://YOUR_PUBLIC_MLFLOW_URL (not the K8s DNS)
```

> **Why this matters:** All SkyPilot YAML defaults are now `MLFLOW_TRACKING_URI: ""`
> (empty). The Airflow DAG always injects the correct value via `task.update_envs()`.
> If the env var is not set on the Airflow pod, the injected value is empty and every
> external VM silently runs without experiment tracking.

### 3. Verify App → Airflow connectivity

```bash
curl -s http://airflow.localhost/api/v2/health | python3 -m json.tool
# Expected: {"metadatabase": {"status": "healthy"}, "scheduler": {"status": "healthy"}}
```

### 4. Verify App → MLflow connectivity (in-cluster)

```bash
kubectl exec -n airflow deploy/my-airflow-api-server -- \
  python3 -c "
import requests
r = requests.get('http://my-mlflow.ray.svc.cluster.local:80/health')
print(r.status_code)
"
# Expected: 200
```

---

## Phase A: Docker Image Build (AWS multi-node only)

> **Skip this phase if you are only using RunPod or Vast.ai.**
> Those providers use native base templates (`runpod/pytorch:...`, `vastai/pytorch:...`) pulled
> directly at VM provisioning time — no custom images needed.
>
> **Required only for:**
> - Tabular multi-node training on AWS (`ray-gpu-multinode-aws.yaml`)
> - LLM fine-tuning on AWS (`ray-llm-training-aws.yaml`)
> - vLLM multi-node serving on AWS (`ray-vllm-multinode-serving.yaml`)

### A1. Prerequisites

```bash
# You must be logged in to DockerHub
docker login
# You need NVIDIA Container Runtime if you want to run GPU smoke tests locally
# (optional — import-only smoke tests work without it)
```

### A2. Build and push both images

```bash
cd k3s/kuberay/gpu
chmod +x build.sh

# Build + push both (takes 10-20 min on first build; subsequent builds use layer cache)
./build.sh

# Or selectively:
./build.sh train   # ray-train:2.53.0 only (tabular)
./build.sh llm     # ray-llm:2.53.0 only (LLM + vLLM)

# Build without pushing (local validation only):
./build.sh --no-push
```

Expected output ends with:
```
Done. Images:
takenking9879/ray-train:2.53.0   ...GB   ...
takenking9879/ray-llm:2.53.0     ...GB   ...
```

### A3. Import smoke tests (no GPU needed)

Verify all critical packages are importable and versions are correct:

```bash
# Tabular training image
docker run --rm takenking9879/ray-train:2.53.0 \
  python3 -c "import ray, torch, deepspeed, xgboost, mlflow; \
  print('ray:', ray.__version__); \
  print('torch:', torch.__version__); \
  print('deepspeed:', deepspeed.__version__); \
  print('xgboost:', xgboost.__version__)"
# Expected: ray 2.53.0, torch 2.10.x, deepspeed 0.18.x, xgboost 3.x

# LLM / vLLM image
docker run --rm takenking9879/ray-llm:2.53.0 \
  python3 -c "import ray, torch, transformers, peft, trl, vllm; \
  print('ray:', ray.__version__); \
  print('torch:', torch.__version__); \
  print('vllm:', vllm.__version__); \
  print('transformers:', transformers.__version__)"
# Expected: ray 2.53.0, torch 2.10.x, vllm 0.18.x, transformers 4.46.x
```

**Common failure:** `ModuleNotFoundError: deepspeed` on ray-train — means `requirements_gpu.txt`
is missing the `deepspeed==0.18.8` line. Add it, then rebuild.

### A4. CUDA smoke test (requires NVIDIA runtime)

```bash
docker run --rm --gpus all takenking9879/ray-train:2.53.0 \
  python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA!'; \
  print('CUDA:', torch.version.cuda, '| Device:', torch.cuda.get_device_name(0))"
# Expected: CUDA: 12.8.x | Device: NVIDIA ...

docker run --rm --gpus all takenking9879/ray-llm:2.53.0 \
  python3 -c "import torch; assert torch.cuda.is_available(); \
  print('CUDA:', torch.version.cuda)"
```

### A5. Verify images are on DockerHub

```bash
docker pull takenking9879/ray-train:2.53.0 && echo "ray-train OK"
docker pull takenking9879/ray-llm:2.53.0   && echo "ray-llm OK"
```

These must succeed from any network (not just your local machine) because SkyPilot pulls them
on the AWS worker VMs.

---

## Phase B: Internal Integration Tests (Docker Desktop only, no cloud cost)

### B1: Preprocessing pipeline via App

1. Open App → **Processing** tab
2. Select dataset `network_traffic`, click Submit
3. Watch Airflow UI → `preprocessing_pipeline` DAG → wait for **SUCCEEDED**
4. Verify the params file was written to S3:
   ```bash
   aws s3 ls s3://k8s-mlops-platform-bucket/runs/preprocessing/ --recursive | head -5
   # Expected: params_preprocess.yaml at runs/preprocessing/{pre-run-id}/params_preprocess.yaml
   ```

### B2: Preprocessing run appears in Run Pipeline dropdown

1. Open App → **Run Pipeline** tab
2. Click the preprocessing run ID dropdown
3. Expected: the run from B1 appears
4. If missing: check `GET /api/v2/processing-runs/ids` directly:
   ```bash
   curl http://your-app/api/v2/processing-runs/ids
   ```

### B3: Prometheus target validation

```bash
kubectl port-forward svc/prometheus-stack-kube-prom-prometheus -n monitoring 9090:9090 &
curl -s http://localhost:9090/api/v1/targets \
  | python3 -m json.tool \
  | grep -E '"health"|"job"' | head -30
# Expected: ray-training, kafka, kube-state-metrics show health="up"
```

---

## Phase C: Cloud Smoke Tests (~$0–1, no GPU)

Run these before any GPU training to confirm network paths work.

### C1: SkyPilot hello-sky (cheapest validation)

```bash
# From Airflow pod (or local with sky configured):
sky launch k3s/sky/hello-sky.yaml --name smoke-test --yes
sky logs smoke-test
# Expected: "Hello from SkyPilot!" within 3 minutes
sky down smoke-test --yes
```

**What this validates:** SkyPilot auth, cloud provider API keys, SSH key provisioning.
**Common failure:** `~/.sky/config.yaml` missing on Airflow pod, or RunPod API key not set.

### C2: MLflow reachability from external VM

```bash
# While smoke-test cluster is still up (before sky down):
sky exec smoke-test -- python3 -c \
  "import urllib.request; print(urllib.request.urlopen('${MLFLOW_EXTERNAL_URL}/health').read())"
# Expected: b'{"status":"OK"}' or b'OK'
sky down smoke-test --yes
```

**Common failure:** `MLFLOW_EXTERNAL_URL` not set, or MLflow has no external Ingress.
Fix: expose MLflow with `kubectl port-forward svc/my-mlflow -n ray 5000:80` for local
testing, or set up an Nginx Ingress pointing to `my-mlflow.ray.svc.cluster.local`.

### C3: S3 + Iceberg connectivity from external VM

```bash
sky launch k3s/sky/hello-sky.yaml --name s3-test --yes
sky exec s3-test -- python3 - <<'EOF'
import boto3, os
s3 = boto3.client('s3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name='us-east-2')
resp = s3.list_objects_v2(Bucket='k8s-mlops-platform-bucket', Prefix='runs/', MaxKeys=3)
print('S3 OK:', [o['Key'] for o in resp.get('Contents', [])])
EOF
sky down s3-test --yes
```

---

## Phase D: Full App-Driven Pipeline Tests

### Case 1: Tabular Training via Run Pipeline (Most Important)

**Goal:** Validate the full lineage chain
`App → preprocessing run → params_training.yaml → Airflow → SkyPilot → Ray → MLflow`

**Steps:**

1. Complete Phase B1 (you need a preprocessing run to exist)
2. Open App → **Run Pipeline** tab
3. Select the preprocessing run from B1 in the dropdown
4. Set: `model_type=ann`, `use_gpu=true`, `providers=[runpod]`, `prefer_spot=true`
5. Click **Submit**
6. In **Airflow UI** → `training_pipeline_skypilot` DAG → click the new run → **Conf** tab
   Verify these keys are present:
   - `train_params_s3_path` — must be non-empty (e.g. `s3://k8s-mlops-platform-bucket/runs/training/...`)
   - `preprocess_run_id` — must match the run you selected
   - `resource_constraints` — must be a dict
7. Download and inspect the params file:
   ```bash
   aws s3 cp $(airflow-conf-train_params_s3_path) - | python3 -c \
     "import sys,yaml; d=yaml.safe_load(sys.stdin); print(list(d.keys()))"
   # Expected keys: run_metadata, execution, lineage, kuberay, hyperparams, iceberg_tables
   ```
8. Wait for DAG **SUCCEEDED** (10–20 min for ANN on RunPod RTX)
9. In **MLflow UI** → search runs by `train_run_id` tag → verify:
   - `val_accuracy > 0.70`
   - `instance_type = spot`
   - `estimated_cost_usd > 0`
   - Model artifact uploaded (Artifacts tab shows model file)

**Repeat for `ssm` and `bae`:** Same steps. Expected `val_accuracy > 0.65`.

---

### Case 2: XGBoost GPU Training

**Goal:** Confirm the XGBoost `device="cuda"` fix works (was silently training on CPU).

1. Same as Case 1 but: `model_type=xgboost`, `use_gpu=true`
2. In the SkyPilot VM logs (visible in Airflow task logs), look for XGBoost output
   containing `device=cuda` in the params
3. Optional GPU utilization check while training:
   ```bash
   sky exec {job_cluster_name} -- nvidia-smi dmon -s u -d 5
   # GPU util column (sm%) should show >0% during training
   ```
4. MLflow: `instance_type=spot`, training should be faster than CPU baseline
5. **Optional baseline comparison:** Run again with `use_gpu=false` on a CPU instance,
   compare training time in MLflow (`start_time` vs `end_time`)

---

### Case 3: Tabular Training via Launch Wizard

**Goal:** Validate `jobs.py` fixes — Airflow API v2, JWT auth, `train_params_s3_path` generation.

1. Open App → **Launch** tab
2. **Step 1:** Model type = `ssm`, job type = Training
3. **Step 2:** Providers = RunPod, prefer_spot = true, GPU count = 1
4. **Step 3:** Enter `preprocess_run_id` from Case 1 → click **Get Recommendation**
   - Expected: orchestration badge shows **Ray Train**, cost estimate appears,
     YAML preview is expandable
   - If HTTP error: open browser devtools → Network tab → check the dry_run request response
5. **Step 4:** Click **Launch**
   - Expected: response contains `job_ids.training` = a `dag_run_id`
6. In Airflow: same validation as Case 1, step 6
7. **Test without preprocess_run_id:** Leave it empty, provide:
   - `dataset=network_traffic`
   - `processed_table=iceberg.processed.network_traffic_abc123` (use a real table name)
   - Job should still launch with a minimal params file (no lineage chain, but valid)

---

### Case 4: LLM Fine-tuning via Launch Wizard

**Prerequisites:** Use `Qwen/Qwen2.5-7B` (ungated, no HF token needed). Requires a GPU
with at least 16 GB VRAM (e.g., A40 on RunPod).

1. Open App → **Launch** tab
2. **Step 1:** Model type = `llm`, model ID = `Qwen/Qwen2.5-7B`, job type = Training
3. **Step 2:** Providers = RunPod, GPU with ≥16 GB VRAM
4. **Step 3:** Dry-run → orchestration should show **HF+TRL+DeepSpeed**
5. **Step 4:** Launch
6. **Airflow** → `llm_training_pipeline` DAG → verify conf has:
   - `llm_model_id: "Qwen/Qwen2.5-7B"`
   - `max_steps` (int)
   - `lora_enabled` (bool)
7. SkyPilot VM logs → look for:
   - `Downloading model weights` (HuggingFace download)
   - `Training step X/Y loss=Z`
8. After completion, check S3 for checkpoint:
   ```bash
   aws s3 ls s3://k8s-mlops-platform-bucket/runs/llm-training/ --recursive | head -5
   ```

---

### Case 5: vLLM Serving (single node)

**Goal:** Validate `serving_configs.py` → `vllm_serving_dag.py` → endpoint registration chain.

1. Open App → **Serving** tab
2. Fill in:
   - Model ID: `Qwen/Qwen2.5-7B`
   - Port: `8000`
   - No HF token needed for this model
3. Click **Deploy vLLM**
4. **Airflow** → `vllm_serving_pipeline` DAG → watch `wait_for_endpoint` task
   (it polls `/health` every 20 seconds — normal to see retries)
5. When DAG shows **SUCCEEDED**, verify endpoint was registered:
   ```bash
   curl -s "http://your-app/api/v2/serving-configs/{serve_run_id}/endpoint"
   # Expected: {"endpoint_url": "http://CLUSTER_IP:8000", "model_id": "Qwen/Qwen2.5-7B", "status": "healthy"}
   ```
6. Test inference:
   ```bash
   curl -X POST http://CLUSTER_IP:8000/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen/Qwen2.5-7B","prompt":"Once upon a time","max_tokens":20}'
   # Expected: JSON with "choices[0].text" containing generated text
   ```
7. Verify auto-terminate safety net: after 60 minutes of idle, the cluster should disappear:
   ```bash
   sky status | grep vllm
   # After 60 min idle: cluster should be gone (idle_minutes_to_autostop: 60 is set)
   ```

---

### Case 6: Full Lineage Chain (Critical Correctness Test)

**Goal:** Prove the complete chain:
`preprocessing run → params_preprocess → params_training → serving config → RayServe deploy`

1. After Case 1 completes, note the `train_run_id` from MLflow
2. Open App → **Serving** tab → "Create serving config for trained tabular model"
3. Enter `train_run_id` → click **Create**
   - Backend reads `runs/training/{train_run_id}/params_training.yaml`
   - Must find `lineage.dataset` — if 400 error: the params file is missing the lineage block
     (download the file and inspect: `aws s3 cp s3://bucket/runs/training/{id}/params_training.yaml -`)
4. Click **Deploy** → Airflow `serving_pipeline` DAG
   - MLflow model should be promoted to `champion` alias
   - RayService in K8s updated with new `params_s3_path`
5. Send a test inference request:
   ```bash
   curl -X POST http://model-serving.localhost/infer \
     -H 'Content-Type: application/json' \
     -d '{"data":[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]]}'
   # Expected: {"prediction": [N], "class_label": "..."}
   ```

---

## Quick Validation Checklist

Run through this before every cloud test session:

- [ ] `pytest tests/test_model_upload.py tests/test_routers.py -v` → 27 passed
- [ ] `MLFLOW_TRACKING_URI` is a **public URL** on the Airflow pod (not K8s DNS)
- [ ] AWS credentials configured in Airflow environment
- [ ] `sky check` shows at least one cloud provider configured
- [ ] `sky launch k3s/sky/hello-sky.yaml --yes` completes in < 5 minutes
- [ ] *(AWS only)* `docker pull takenking9879/ray-train:2.53.0` succeeds — images are on DockerHub

After each cloud test session:

- [ ] `sky status` shows no orphaned clusters (any leftover clusters cost money)
- [ ] `sky down --all --yes` if anything is still running
- [ ] Check AWS billing dashboard for unexpected charges

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `KeyError: 'train_params_s3_path'` in Airflow | dag_conf missing key | jobs.py `_upload_training_params()` must run before DAG trigger; check S3 for the file |
| No MLflow run after training | K8s DNS fallback active | Set `MLFLOW_TRACKING_URI` on Airflow pod to a public URL |
| XGBoost GPU util = 0% | Missing `device="cuda"` | Fixed in `xgboost_utils.py` — verify the env var `USE_GPU=true` is passed to the VM |
| Preprocessing run missing from dropdown | S3 path mismatch | Check `runs/preprocessing/` prefix exists in S3; check `GET /api/v2/processing-runs/ids` |
| Airflow 401/403 from jobs.py | Old basic auth | Fixed — now uses JWT via `_call_airflow` from runs.py |
| Airflow 404 on DAG trigger | API v1 path used | Fixed — all calls use `/api/v2/`; if you see 404, check the full URL in Airflow task logs |
| vLLM cluster never stops | No idle autostop | Fixed — `idle_minutes_to_autostop: 60` added to both serving YAMLs |
| `400 Could not resolve dataset` on serving | Missing lineage block | Fixed — `_upload_training_params` always writes lineage; re-trigger training to regenerate params |
| hello-sky hangs > 10 min | Cloud auth or quota issue | Run `sky check`, verify API keys; check RunPod quota / spot availability |
| SkyPilot VM can't reach S3 | AWS creds not injected | Verify `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are in Airflow env vars |
| AWS VM: `docker pull` fails for ray-train/ray-llm | Images not pushed to DockerHub | Run `./k3s/kuberay/gpu/build.sh` and push; then verify with `docker pull takenking9879/ray-train:2.53.0` |
| `ModuleNotFoundError: deepspeed` on AWS job | Missing `deepspeed` in requirements_gpu.txt | Add `deepspeed==0.18.8` to `k3s/kuberay/gpu/requirements_gpu.txt`, rebuild with `./build.sh train` |
| RunPod/Vast: CUDA version mismatch | Wrong base image or torch reinstalled | Never reinstall torch on provider templates — `requirements_tabular_runtime.txt` and `requirements_llm_runtime.txt` must exclude torch |
| Wrong YAML loaded for provider | `resource_constraints` missing from dag_conf | Fixed in `jobs.py` — check Airflow conf tab shows `resource_constraints` key |
