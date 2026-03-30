# k3s/ — Key Elements

## DAG Inventory

| DAG ID | File | Status | What it does |
|--------|------|--------|-------------|
| `preprocessing_pipeline` | `preprocessing_dag.py` | ACTIVE | Submit + poll SparkApplication; cleanup in finally |
| `training_pipeline` | `training_dag.py` | ACTIVE | Submit + poll RayJob (KubeRay local); cleanup in finally |
| `training_pipeline_skypilot` | `training_dag_skypilot.py` | ACTIVE | **Single-pod lifecycle**: one KubernetesPodOperator runs `sky_runner.py run-training` which does submit → poll → cancel-on-failure within the same pod (same SkyPilot local API server). Safety-net cleanup task on failure. Resources tunable via `SKY_SUBMIT_*` env vars. |
| `llm_training_pipeline` | `llm_training_dag.py` | ACTIVE | **Single-pod lifecycle**: one KubernetesPodOperator runs `sky_runner.py run-llm` (submit → poll → cancel-on-failure). SkyPilot LLM fine-tuning (TRL + LoRA + DeepSpeed ZeRO). |
| `full_ml_pipeline` | `full_pipeline_dag.py` | ACTIVE | preprocessing_pipeline tasks → training_pipeline tasks (sequential chain) |
| `ml_pipeline` | `ml_pipeline_dag.py` | ACTIVE | Flexible mode: preprocessing_only / training_only / full; UI-triggered |
| `vllm_serving_pipeline` | `ray_vllm_serving_dag.py` | ACTIVE | launch_cluster → wait_for_endpoint → register_endpoint (S3); `sky.launch(detach_run=True)` |
| `serving_pipeline` | `serving_dag.py` | ACTIVE | promote_to_champion → patch_ray_service_config → branch_on_serving_mode [ray_only / kafka] |
| `model_promotion_workflow` | `promotion_dag.py` | ACTIVE | Manual/auto MLflow alias assignment; reads f1_score, sets @champion/@challenger |
| *(empty)* | `dag.py` | **LEGACY** | Empty stub; no functionality |

### serving_pipeline detail
Tasks:
1. `promote_to_champion` — MLflow: search by train_run_id tag → set @champion alias
2. `patch_ray_service_config` — K8s: GET/PATCH RayService to inject `PARAMS_SERVING_S3_PATH` into all apps' runtime_env
3. `branch_on_serving_mode` — BranchPythonOperator: `ray_only` → `skip_kafka`; `kafka` → validate_raw_schema
4. (kafka branch): `validate_raw_schema` → `delete_existing_spark_connector` → `submit_spark_connector` → `poll_spark_connector_running`
5. Spark connector NOT deleted on success (streaming must stay running)

---

## SkyPilot YAML Inventory

| File | Provider | Kind | Status |
|------|----------|------|--------|
| `ray-gpu-training-runpod.yaml` | RunPod | train | ACTIVE |
| `ray-gpu-training-runpod-smoke.yaml` | RunPod | utility/smoke | ACTIVE (manual validation of setup + Ray bootstrap; no full training) |
| `ray-gpu-training-vast.yaml` | Vast.ai | train | ACTIVE |
| `ray-gpu-multinode-aws.yaml` | AWS | train_multi | ACTIVE |
| `ray-llm-training-runpod.yaml` | RunPod | llm | ACTIVE |
| `ray-llm-training-vast.yaml` | Vast.ai | llm | ACTIVE |
| `ray-llm-training-aws.yaml` | AWS | llm | ACTIVE |
| `vllm-serving-runpod.yaml` | RunPod | vllm | ACTIVE |
| `vllm-serving-vast.yaml` | Vast.ai | vllm | ACTIVE |
| `ray-vllm-multinode-serving.yaml` | AWS | vllm_multi | ACTIVE |
| `sky-runner-pod.yaml` | K8s | utility | ACTIVE (reference/template only; KubernetesPodOperator runtime spec is defined in DAG files) |
| `ray-gpu-training.yaml` | generic | train | **LEGACY** |
| `ray-llm-training.yaml` | generic | llm | **LEGACY** |
| `vllm-serving.yaml` | generic | vllm | **LEGACY** |
| `hello-sky.yaml` | test | — | **LEGACY** |
| `vast-test.yaml` | test | — | **LEGACY** |

Routing table (kind, provider) → YAML: defined in `app/backend/services/job_builder.py` and `k3s/airflow/dags/sky_runner.py`.

Sky runner runtime notes:
- `k3s/sky/docker-entrypoint.sh` writes provider credentials for RunPod and Vast (`VAST_API_KEY` or `VASTAI_API_KEY`).
- It also writes/merges `~/.sky/config.yaml` to enforce `jobs.controller.resources.disk_size` (default 30 GB, clamped to 40 GB for RunPod compatibility).
- **Single-pod lifecycle** (`run-training`, `run-llm`): submit + poll run in the same pod, reusing one SkyPilot local API server. Legacy split-phase commands (`submit-training`, `poll-training`, `submit-llm`, `poll-llm`) are kept for debugging but no longer used by DAGs.
- Managed jobs polling uses `sky.jobs.queue(version=2)` with `all_users=True`.
- All provider-native (RunPod/Vast) YAMLs use a **probe-based venv pattern**: setup probes for the image Python with torch+CUDA, creates `/opt/ml-platform-runtime` via `${PYTHON_BIN} -m venv --system-site-packages` (inherits provider torch), writes the venv Python path to `<APP_HOME>/runtime_python.txt`, and installs deps via `${RUNTIME_PYTHON} -m pip`. Run reads the file and uses explicit binary paths (`"${PYTHON_BIN}"`, `"${RAY_BIN}"`, `"${VLLM_BIN}"`) — no PATH-dependent bare commands.
- For GPU training YAMLs, DeepSpeed is installed only when `USE_DEEPSPEED=true`; default skips it. `job_builder.py` always overrides this env from `config.use_deepspeed` — the YAML default is never used at runtime. `LaunchWizardPage` sets `use_deepspeed=true, deepspeed_stage=3` for LLM jobs and `false/1` for tabular, matching the YAML defaults.
- On managed job terminal failure, `k3s/sky/sky_runner.py` now includes queue-provided diagnostics (`failure_reason`, `failure_type`, `cluster_name`, etc.) in raised errors.
- `rc['gpu_fallbacks']` (list of `{infra, accelerators, use_spot}`) bypasses the auto-catalog selector and injects the list directly into `resources.any_of`. Set from `ResourceConstraints.gpu_fallbacks` in the frontend.

---

## K8s Manifests

| File | Purpose | Status |
|------|---------|--------|
| `k3s/spark/spark-application.yaml` | Base SparkApplication (preprocessing); patched per run | ACTIVE |
| `k3s/spark/inference/spark-kafka-application.yaml` | Spark Kafka streaming connector (serving kafka branch) | ACTIVE |
| `k3s/kuberay/kuberay-job.yaml` | Base RayJob (local training); patched per run | ACTIVE |
| `k3s/kuberay/serving/rayservice-model-serving.yaml` | RayService (production serving); patched by serving_dag | ACTIVE |
| `k3s/kuberay/kuberay-job-gpu.yaml` | GPU-specific RayJob variant | **LEGACY** |

---

## k8s_helpers.py

Function/Class: utility functions
File: `k3s/airflow/k8s_helpers.py`

Functions:
- `delete_spark_app(client, namespace, name)` — idempotent SparkApplication delete; swallows 404
- `delete_ray_job(client, namespace, name)` — idempotent RayJob delete; swallows 404
- `patch_ray_service_config(client, namespace, ray_service_name, params_s3_path)` — inject PARAMS_SERVING_S3_PATH into RayService serveConfigV2 (all apps' runtime_env.env_vars)
- `k8s_name(run_id, prefix, max_len=63)` — RFC-1123 compliant K8s name; preserves unique hash suffix on truncation (critical: RayJob names ≤ 47 chars)
