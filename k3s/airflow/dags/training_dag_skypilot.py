"""Training Pipeline DAG — SkyPilot managed-jobs-based training run.

Drop-in companion to training_dag.py. Accepts the same dag_run.conf keys and
replaces the KubeRay CRD submission path with sky.jobs.launch() → RunPod (or
any cloud). No Virtual Kubelet, no Tailscale, no Kubernetes CustomObjectsApi.

Uses sky.jobs (managed jobs) instead of sky.launch():
  - Automatically handles spot preemptions: re-provisions + restarts the job
  - Ephemeral cluster is auto-cleaned after the job completes (no sky.down needed)
  - Use sky.jobs.queue() (not sky.queue()) to poll job status

dag_run.conf (same as training_pipeline):
    train_run_id            : str        — unique training run ID
    preprocess_run_id       : str        — referenced preprocessing run ID
    dataset                 : str        — canonical dataset name
    processed_table         : str        — Iceberg processed table to train on
    dsl_s3_path             : str        — S3 URI to DSL YAML
    train_params_s3_path    : str        — S3 URI of params_training.yaml
    model_type              : str        — "xgboost" → CPU  |  "pytorch" → GPU
    resource_constraints    : dict|None  — optional GPUSelectorService constraints;
                                           when present, any_of is built dynamically
                                           from the real-time catalog instead of
                                           using the static YAML any_of entries

Tasks:
    1. submit_sky_job  — load sky YAML, inject per-run env vars, launch managed job, push job_name to XCom
    2. poll_sky_job    — poll sky.jobs.queue() every 30s; cancels job on timeout/failure

Requirements in Airflow image:
    pip install "skypilot[runpod]"

SkyPilot config on the Airflow worker host (NOT in the repo):
    ~/.sky/config.yaml:
        runpod:
          api_key: rpa_XXXXXXXXXXXXXXXXXX

AWS credentials for S3 (one of):
    A) ~/.aws/credentials on the Airflow worker
    B) env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY passed via task.update_envs()
    C) IAM role attached to the Airflow worker pod
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── Constants ────────────────────────────────────────────────────────────────

# SkyPilot YAML paths — override via env vars in docker-compose / Helm values
SKY_YAML_CPU = Path(
    os.getenv(
        "SKY_TRAINING_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-training.yaml",
    )
)
SKY_YAML_GPU = Path(
    os.getenv(
        "SKY_TRAINING_GPU_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-gpu-training.yaml",
    )
)
SKY_YAML_GPU_MULTINODE = Path(
    os.getenv(
        "SKY_TRAINING_GPU_MULTINODE_YAML",
        "/opt/airflow/dags/repo/k3s/sky/ray-gpu-multinode-aws.yaml",
    )
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://my-mlflow.ray.svc.cluster.local:80",
)
SKY_TIMEOUT_SECONDS = int(os.getenv("SKY_TIMEOUT_SECONDS", "7200"))
SKY_IDLE_MINUTES = int(os.getenv("SKY_IDLE_MINUTES_AUTOSTOP", "15"))

_DAGS_DIR = Path(__file__).parent.parent  # k3s/airflow/


# ─── Helpers ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root


def _load_task_with_resources(
    yaml_path: str,
    train_run_id: str,
    resource_constraints_conf: dict | None,
    use_gpu: bool,
    num_nodes: int = 1,
):
    """Load a SkyPilot Task from YAML, optionally replacing any_of with a
    dynamically generated spot-first list from the live GPU catalog.

    Falls back to static YAML when:
      - resource_constraints_conf is None / not a GPU job
      - catalog query returns no offers
      - selector produces an empty any_of
    """
    import sky
    import yaml as _yaml

    if not (use_gpu and resource_constraints_conf):
        return sky.Task.from_yaml(yaml_path)

    # Import catalog / selector from src (project root must be on path)
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    try:
        import dataclasses
        from src.services.gpu_catalog import GPUCatalogService
        from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

        _valid_fields = {f.name for f in dataclasses.fields(ResourceConstraints)}
        merged = {k: v for k, v in resource_constraints_conf.items() if k in _valid_fields}
        # num_nodes from dag_run.conf overrides whatever was in resource_constraints
        if "num_nodes" in _valid_fields:
            merged["num_nodes"] = num_nodes
        constraints = ResourceConstraints(**merged)

        offers = GPUCatalogService().query_availability(
            providers=constraints.providers,
            min_vram_gb=constraints.min_vram_gb,
            gpu_types=constraints.gpu_types,
        )
        result = GPUSelectorService().select_providers(constraints, offers)

        if not result.any_of:
            print("Dynamic selector returned empty list — falling back to static YAML")
            return sky.Task.from_yaml(yaml_path)

        # Load base YAML as dict, replace only the any_of block
        with open(yaml_path) as fh:
            sky_conf = _yaml.safe_load(fh)

        sky_conf.setdefault("resources", {})["any_of"] = result.any_of
        # Remove top-level keys that are not valid in sky jobs launch
        sky_conf["resources"].pop("infra", None)

        tmp_path = f"/tmp/sky_job_{train_run_id}.yaml"
        with open(tmp_path, "w") as fh:
            _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

        print(
            f"Dynamic any_of: {len(result.any_of)} entries "
            f"({result.spot_entries} spot, {result.ondemand_entries} on-demand)"
        )
        return sky.Task.from_yaml(tmp_path)

    except Exception as exc:
        print(f"[warn] Dynamic resource selection failed ({exc}) — using static YAML")
        return sky.Task.from_yaml(yaml_path)


# ─── Tasks ───────────────────────────────────────────────────────────────────


def _submit_sky_job(**context):
    """Load a SkyPilot task YAML, inject per-run env vars, and launch a managed job.

    Routing:
        model_type == "pytorch"  → ray-gpu-training.yaml  (RunPod GPU, spot-first)
        model_type == "xgboost"  → ray-training.yaml       (RunPod CPU)

    Uses sky.jobs.launch() (managed jobs):
      - Spot preemptions are handled automatically: new node provisioned + job restarted
      - Cluster is ephemeral — auto-cleaned after completion (no sky.down() needed)

    Pushes sky_job_name to XCom so poll_sky_job can monitor it.
    """
    import sky

    sys.path.insert(0, str(_DAGS_DIR))
    from k8s_helpers import k8s_name  # reuse RFC-1123 slug utility for job names

    conf = context["dag_run"].conf or {}
    train_run_id      = conf["train_run_id"]
    preprocess_run_id = conf["preprocess_run_id"]
    processed_table   = conf.get("processed_table", "")
    model_type        = conf.get("model_type", "xgboost")
    params_s3_path    = conf["train_params_s3_path"]
    num_nodes         = int(conf.get("num_nodes", 1))

    # Job name: max 40 chars, RFC-1123 slug (SkyPilot managed job naming requirement)
    job_name = k8s_name(train_run_id, "sky", max_len=40)

    # Select YAML based on model type and node count
    #   pytorch + num_nodes > 1  → multi-node AWS  (EFA, DeepSpeed)
    #   pytorch + num_nodes == 1 → single-node GPU  (RunPod spot-first)
    #   everything else          → CPU              (RunPod CPU)
    use_gpu = model_type in ("pytorch", "ssm", "bae")
    if use_gpu and num_nodes > 1:
        yaml_path = str(SKY_YAML_GPU_MULTINODE)
    elif use_gpu:
        yaml_path = str(SKY_YAML_GPU)
    else:
        yaml_path = str(SKY_YAML_CPU)
    print(f"Using SkyPilot YAML: {yaml_path}  (gpu={use_gpu}, num_nodes={num_nodes})")

    # ── Dynamic any_of from resource_constraints (Phase 3) ─────────────────────
    # When resource_constraints is provided in dag_run.conf, we query the live
    # GPU catalog and replace the static any_of in the YAML with a dynamically
    # ranked spot-first list.  Falls back to static YAML if catalog returns nothing.
    resource_constraints_conf = conf.get("resource_constraints")
    task = _load_task_with_resources(
        yaml_path, train_run_id, resource_constraints_conf, use_gpu,
        num_nodes=num_nodes,
    )
    # ─────────────────────────────────────────────────────────────────────────────

    # Inject all per-run parameters as environment variables.
    # These override the placeholder defaults defined in the YAML envs: section.
    envs: dict = {
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "PARAMS_S3_PATH":      params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    }
    # For multi-node jobs, override NUM_WORKERS to match actual GPU count
    if num_nodes > 1:
        gpus_per_node = 8  # p4d/p3dn; adjust via OVERRIDE_GPUS_PER_NODE if needed
        envs["NUM_WORKERS"] = str(num_nodes * gpus_per_node)

    task.update_envs(envs)

    # Launch as a managed job (handles spot preemption automatically).
    # sky.jobs.launch() returns a RequestId; stream_and_get() blocks until the
    # job is accepted by the jobs controller (fast — does NOT wait for completion).
    # The controller then manages spot recovery, provisioning, etc. autonomously.
    # poll_sky_job polls sky.jobs.queue() to track completion.
    sky.stream_and_get(sky.jobs.launch(
        task,
        name=job_name,
    ))
    print(f"SkyPilot managed job launched: {job_name}")

    # Pass job name downstream for polling
    context["ti"].xcom_push(key="sky_job_name", value=job_name)


def _poll_sky_job(**context):
    """Poll sky.jobs.queue() every 30s until the managed job reaches a terminal state.

    Terminal states: SUCCEEDED → return normally; FAILED / CANCELLED → raise.
    On timeout or failure, cancels the managed job so it doesn't keep running.
    No sky.down() needed — managed job clusters are auto-cleaned on completion.
    """
    import sky
    import time

    job_name = context["ti"].xcom_pull(
        task_ids="submit_sky_job", key="sky_job_name"
    )
    print(f"Polling managed job: {job_name}")

    interval = 30
    elapsed  = 0

    try:
        while elapsed < SKY_TIMEOUT_SECONDS:
            time.sleep(interval)
            elapsed += interval

            try:
                # refresh=False: faster poll (uses cached controller state)
                # sky.jobs.queue() returns a RequestId; sky.get() resolves it.
                all_jobs = sky.get(sky.jobs.queue(refresh=False))
            except Exception as exc:
                print(f"[{elapsed}s] sky.jobs.queue() error: {exc} — retrying...")
                continue

            # Filter to our job by name
            our_jobs = [j for j in all_jobs if j.get("job_name") == job_name]
            if not our_jobs:
                print(f"[{elapsed}s] Job '{job_name}' not visible yet (provisioning...)")
                continue

            # Most recent entry for this job name
            last_job   = our_jobs[-1]
            job_status = str(last_job.get("status", ""))
            print(f"[{elapsed}s] Managed job '{job_name}' status: {job_status}")

            if "SUCCEEDED" in job_status:
                print("Training completed successfully.")
                return
            if "FAILED" in job_status or "CANCELLED" in job_status:
                raise RuntimeError(
                    f"SkyPilot managed job '{job_name}' ended with status: {job_status}"
                )
            # PENDING / STARTING / RUNNING / RECOVERING → keep polling

        raise RuntimeError(
            f"Managed job '{job_name}' timed out after {SKY_TIMEOUT_SECONDS}s."
        )

    except Exception:
        # Cancel the managed job so it doesn't keep running after Airflow gives up.
        print(f"Cancelling managed job '{job_name}' due to error or timeout.")
        try:
            sky.get(sky.jobs.cancel(name=job_name))
            print(f"Managed job '{job_name}' cancelled.")
        except Exception as exc:
            print(f"Warning: sky.jobs.cancel('{job_name}') failed: {exc}")
        raise


# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline_skypilot",
    description=(
        "SkyPilot managed-jobs training pipeline. "
        "Same dag_run.conf as training_pipeline. "
        "Routes pytorch → GPU spot-first (sky.jobs.launch), xgboost → CPU on RunPod."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "training", "skypilot"],
) as dag:

    submit_sky = PythonOperator(
        task_id="submit_sky_job",
        python_callable=_submit_sky_job,
    )

    poll_sky = PythonOperator(
        task_id="poll_sky_job",
        python_callable=_poll_sky_job,
    )

    submit_sky >> poll_sky
