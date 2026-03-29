#!/usr/bin/env python3
"""SkyPilot Runner — CLI script for Airflow KubernetesPodOperator.

Runs inside takenking9879/sky-runner:0.12.0.
All inputs come from environment variables; return values are written to
/airflow/xcom/return.json so KubernetesPodOperator do_xcom_push=True works.

Commands:
    submit-training     Launch managed tabular training job; pushes job_name (str)
    poll-training       Poll managed training job until terminal state
    cleanup-failed-training-controller
                        Best-effort cleanup: cancel failed training job and
                        tear down SkyPilot jobs controller when safe
    submit-llm          Launch managed LLM fine-tuning job; pushes job_name (str)
    poll-llm            Poll managed LLM job until terminal state
    launch-vllm         Provision persistent single-node vLLM cluster; pushes cluster_info (dict)
    wait-vllm           Poll /health until ready; pushes endpoint_url (str)
    launch-ray-vllm     Provision persistent multi-node Ray+vLLM cluster; pushes cluster_info (dict)
    wait-ray-vllm       Poll /health until ready (multi-node); pushes endpoint_url (str)
    register-endpoint   Write endpoint URL to S3 (shared by both serving types)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# ─── XCom ─────────────────────────────────────────────────────────────────────

_XCOM_DIR = Path("/airflow/xcom")


def _xcom_push(value) -> None:
    _XCOM_DIR.mkdir(parents=True, exist_ok=True)
    (_XCOM_DIR / "return.json").write_text(json.dumps(value))


# ─── Environment helpers ───────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _rc() -> dict | None:
    """Parse RESOURCE_CONSTRAINTS_JSON env var; returns None when absent/empty."""
    raw = _env("RESOURCE_CONSTRAINTS_JSON")
    if not raw or raw in ("null", "{}"):
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _provider(rc: dict | None) -> str:
    providers = (rc or {}).get("providers") or []
    return str(providers[0]).lower() if providers else "runpod"


# ─── YAML routing ─────────────────────────────────────────────────────────────

_SKY_YAML_DIR = Path(os.getenv("SKY_YAML_DIR", "/app/k3s/sky"))

# (kind, provider) → filename
_YAML_MAP: dict[tuple[str, str], str] = {
    ("train",      "runpod"): "ray-gpu-training-runpod.yaml",
    ("train",      "vast"):   "ray-gpu-training-vast.yaml",
    ("train_multi","aws"):    "ray-gpu-multinode-aws.yaml",
    ("llm",        "runpod"): "ray-llm-training-runpod.yaml",
    ("llm",        "vast"):   "ray-llm-training-vast.yaml",
    ("llm",        "aws"):    "ray-llm-training-aws.yaml",
    ("vllm",       "runpod"): "vllm-serving-runpod.yaml",
    ("vllm",       "vast"):   "vllm-serving-vast.yaml",
    ("vllm_multi", "aws"):    "ray-vllm-multinode-serving.yaml",
}


def _yaml_path(kind: str, rc: dict | None) -> str:
    p = _provider(rc)
    filename = _YAML_MAP.get((kind, p)) or _YAML_MAP.get((kind, "runpod"))
    if not filename:
        raise ValueError(f"No YAML mapping for kind={kind!r} provider={p!r}")
    full = _SKY_YAML_DIR / filename
    if not full.exists():
        raise FileNotFoundError(f"SkyPilot YAML not found: {full}")
    return str(full)


# ─── Task loading with optional dynamic resource selection ────────────────────

def _load_task(yaml_path: str, run_id: str, rc: dict | None, prefer_spot: bool = True):
    """Load a SkyPilot Task from yaml_path, optionally replacing any_of with a
    dynamically ranked spot-first list from the GPU catalog, or with an explicit
    user-defined fallback list from rc['gpu_fallbacks'].

    Falls back to the static YAML on any import/catalog error.
    """
    import sky
    import yaml as _yaml

    if not rc:
        return sky.Task.from_yaml(yaml_path)

    # ── Explicit user-defined fallback list ────────────────────────────────
    gpu_fallbacks = rc.get("gpu_fallbacks")
    if gpu_fallbacks and isinstance(gpu_fallbacks, list) and len(gpu_fallbacks) > 0:
        try:
            with open(yaml_path) as fh:
                sky_conf = _yaml.safe_load(fh)
            sky_conf.setdefault("resources", {})["any_of"] = gpu_fallbacks
            sky_conf["resources"].pop("infra", None)
            tmp = f"/tmp/sky_task_{run_id}_fallback.yaml"
            with open(tmp, "w") as fh:
                _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)
            print(f"Using explicit gpu_fallbacks: {len(gpu_fallbacks)} entries (spot-first order)")
            for i, entry in enumerate(gpu_fallbacks):
                print(f"  [{i+1}] infra={entry.get('infra','?')} accel={entry.get('accelerators','?')} spot={entry.get('use_spot','?')}")
            return sky.Task.from_yaml(tmp)
        except Exception as exc:
            print(f"[warn] gpu_fallbacks injection failed ({exc}) — falling back to catalog selector")

    try:
        import dataclasses

        # src/ is at /app/src/ in the image
        _app = Path(__file__).parent
        if str(_app) not in sys.path:
            sys.path.insert(0, str(_app))

        from src.services.gpu_catalog import GPUCatalogService
        from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

        _valid = {f.name for f in dataclasses.fields(ResourceConstraints)}
        merged = {k: v for k, v in rc.items() if k in _valid}
        if not prefer_spot:
            merged["prefer_spot"] = False
        constraints = ResourceConstraints(**merged)

        offers = GPUCatalogService().query_availability(
            providers=constraints.providers,
            min_vram_gb=constraints.min_vram_gb,
            gpu_types=constraints.gpu_types,
        )
        result = GPUSelectorService().select_providers(constraints, offers)

        if not result.any_of:
            print("Dynamic selector returned empty list — using static YAML")
            return sky.Task.from_yaml(yaml_path)

        with open(yaml_path) as fh:
            sky_conf = _yaml.safe_load(fh)
        sky_conf.setdefault("resources", {})["any_of"] = result.any_of
        sky_conf["resources"].pop("infra", None)

        tmp = f"/tmp/sky_task_{run_id}.yaml"
        with open(tmp, "w") as fh:
            _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

        print(
            f"Dynamic any_of: {len(result.any_of)} entries "
            f"({result.spot_entries} spot, {result.ondemand_entries} on-demand)"
        )
        return sky.Task.from_yaml(tmp)

    except Exception as exc:
        print(f"[warn] Dynamic resource selection failed ({exc}) — using static YAML")
        return sky.Task.from_yaml(yaml_path)


# ─── RFC-1123 slug helper ─────────────────────────────────────────────────────

def _k8s_name(name: str, prefix: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9-]", "-", f"{prefix}-{name}".lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:max_len]


# ─── Cluster IP helper ────────────────────────────────────────────────────────

def _get_head_ip(cluster_name: str) -> str | None:
    import sky
    try:
        records = sky.get(sky.status(cluster_names=[cluster_name], refresh=True))
        if not records:
            return None
        handle = records[0].get("handle")
        if handle and hasattr(handle, "external_ips"):
            ips = handle.external_ips()
            if ips:
                return str(ips[0])
    except Exception as exc:
        print(f"[warn] Could not get head IP for '{cluster_name}': {exc}")
    return None


def _is_stream_connection_error(exc: Exception) -> bool:
    """Best-effort detection for transient SkyPilot stream API failures."""
    msg = str(exc)
    return (
        "/api/stream" in msg
        and ("Connection refused" in msg or "Max retries exceeded" in msg)
    )


def _job_record_to_dict(job: object) -> dict:
    """Normalize queue records to plain dicts across SkyPilot versions."""
    if isinstance(job, dict):
        return job

    model_dump = getattr(job, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    legacy_dict = getattr(job, "dict", None)
    if callable(legacy_dict):
        try:
            dumped = legacy_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    return {
        "job_name": str(getattr(job, "job_name", "")),
        "status": str(getattr(job, "status", "")),
    }


def _status_text(status: object) -> str:
    if status is None:
        return ""
    value = getattr(status, "value", None)
    return str(value) if value is not None else str(status)


def _is_terminal_status(status: object) -> bool:
    text = _status_text(status).upper()
    return any(s in text for s in ("SUCCEEDED", "FAILED", "CANCELLED", "CANCELED"))


def _queue_managed_jobs(
    refresh: bool,
    skip_finished: bool = False,
    all_users: bool = True,
) -> list[dict]:
    """Fetch managed jobs via queue(version=2) and normalize records.

    `all_users=True` avoids cross-pod user-hash mismatches in ephemeral
    Airflow KPO pods where ~/.sky state is not shared.
    """
    import sky

    try:
        result = sky.get(
            sky.jobs.queue(
                refresh=refresh,
                skip_finished=skip_finished,
                all_users=all_users,
                version=2,
            ))
    except Exception as exc:
        # SkyPilot may raise this when no job is currently in progress.
        if "No in-progress managed jobs." in str(exc):
            return []
        raise

    records = result[0] if isinstance(result, tuple) else result
    if records is None:
        return []
    return [_job_record_to_dict(job) for job in records]


def _managed_job_visible(
    job_name: str,
    timeout_seconds: int = 120,
    poll_interval_seconds: int = 10,
) -> bool:
    """Return True if a managed job is visible in queue(version=2) within timeout."""

    waited = 0
    while waited <= timeout_seconds:
        try:
            all_jobs = _queue_managed_jobs(refresh=True, skip_finished=False, all_users=True)
            if any(j.get("job_name") == job_name for j in all_jobs):
                return True
        except Exception as queue_exc:
            print(f"[warn] queue visibility check failed: {queue_exc}")

        if waited == timeout_seconds:
            break
        time.sleep(poll_interval_seconds)
        waited += poll_interval_seconds

    return False


def _launch_managed_job_with_stream_fallback(task, job_name: str) -> None:
    """Launch managed job and tolerate transient stream API errors.

    SkyPilot may occasionally fail the local /api/stream follow request even when
    launch succeeded. In that case, validate the job appears in queue before
    failing the task.
    """
    import sky

    request_id = sky.jobs.launch(task, name=job_name)
    try:
        sky.stream_and_get(request_id)
        return
    except Exception as exc:
        if not _is_stream_connection_error(exc):
            raise

        print(f"[warn] stream_and_get failed for '{job_name}': {exc}")
        print("Verifying job visibility in sky.jobs.queue(version=2) before failing...")
        if _managed_job_visible(job_name):
            print("Managed job is visible in queue; continuing despite stream error.")
            return

        raise RuntimeError(
            f"Managed job '{job_name}' launch stream failed and the job is not visible in queue."
        ) from exc


# ─── Operations ───────────────────────────────────────────────────────────────

def submit_training():
    train_run_id      = _env("TRAIN_RUN_ID")
    preprocess_run_id = _env("PREPROCESS_RUN_ID")
    processed_table   = _env("PROCESSED_TABLE")
    model_type        = _env("MODEL_TYPE", "xgboost")
    params_s3_path    = _env("PARAMS_S3_PATH")
    num_nodes         = int(_env("NUM_NODES", "1"))
    mlflow_uri        = _env("MLFLOW_TRACKING_URI")
    rc                = _rc()

    job_name = _k8s_name(train_run_id, "sky")
    use_gpu  = model_type in ("pytorch", "ssm", "bae")
    kind     = "train_multi" if (use_gpu and num_nodes > 1) else ("train" if use_gpu else "train")

    yaml_path = _yaml_path(kind, rc)
    print(f"Using YAML: {yaml_path}  (model={model_type} gpu={use_gpu} nodes={num_nodes})")

    task = _load_task(yaml_path, train_run_id, rc if use_gpu else None)
    envs: dict[str, str] = {
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "PARAMS_S3_PATH":      params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "MLFLOW_TRACKING_URI": mlflow_uri,
    }
    if num_nodes > 1:
        envs["NUM_WORKERS"] = str(num_nodes * 8)
    task.update_envs(envs)

    _launch_managed_job_with_stream_fallback(task, job_name)
    print(f"Managed training job launched: {job_name}")
    _xcom_push(job_name)


def poll_training():
    job_name = _env("SKY_JOB_NAME")
    timeout  = int(_env("SKY_TIMEOUT_SECONDS", "7200"))
    print(f"Polling managed training job: {job_name}")

    interval, elapsed = 30, 0
    try:
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            try:
                all_jobs = _queue_managed_jobs(
                    refresh=(elapsed == interval),
                    skip_finished=False,
                    all_users=True,
                )
            except Exception as exc:
                print(f"[{elapsed}s] sky.jobs.queue(version=2) error: {exc} — retrying...")
                continue

            our = [j for j in all_jobs if j.get("job_name") == job_name]
            if not our:
                print(f"[{elapsed}s] Job '{job_name}' not visible yet (provisioning...)")
                continue

            status = _status_text(our[-1].get("status", ""))
            print(f"[{elapsed}s] Status: {status}")
            if "SUCCEEDED" in status:
                print("Training completed successfully.")
                return
            if "FAILED" in status or "CANCELLED" in status:
                raise RuntimeError(f"Job '{job_name}' ended with status: {status}")

        raise RuntimeError(f"Job '{job_name}' timed out after {timeout}s.")

    except Exception:
        print(f"Cancelling job '{job_name}' due to error/timeout.")
        try:
            import sky
            sky.get(sky.jobs.cancel(name=job_name))
        except Exception as exc:
            print(f"Warning: cancel failed: {exc}")
        raise


def submit_llm():
    run_id     = _env("LLM_TRAIN_RUN_ID")
    model_id   = _env("LLM_MODEL_ID")
    dataset_s3 = _env("TRAIN_DATASET_S3")
    hf_token   = _env("HF_TOKEN")
    max_steps  = int(_env("MAX_STEPS", "500"))
    save_steps = int(_env("SAVE_STEPS", "100"))
    lora_on    = _env("LORA_ENABLED", "true")
    lora_rank  = int(_env("LORA_RANK", "8"))
    ds_stage   = int(_env("DEEPSPEED_STAGE", "3"))
    mlflow_uri = _env("MLFLOW_TRACKING_URI")
    rc         = _rc()

    job_name  = _k8s_name(run_id, "llm")
    yaml_path = _yaml_path("llm", rc)
    print(f"Using LLM YAML: {yaml_path}")

    task = _load_task(yaml_path, run_id, rc)
    task.update_envs({
        "LLM_TRAIN_RUN_ID":    run_id,
        "LLM_MODEL_ID":        model_id,
        "HF_TOKEN":            hf_token,
        "TRAIN_DATASET_S3":    dataset_s3,
        "MAX_STEPS":           str(max_steps),
        "SAVE_STEPS":          str(save_steps),
        "LORA_ENABLED":        lora_on,
        "LORA_RANK":           str(lora_rank),
        "DEEPSPEED_STAGE":     str(ds_stage),
        "MLFLOW_TRACKING_URI": mlflow_uri,
    })

    _launch_managed_job_with_stream_fallback(task, job_name)
    print(f"LLM managed job launched: {job_name}")
    _xcom_push(job_name)


def poll_llm():
    job_name = _env("SKY_JOB_NAME")
    timeout  = int(_env("SKY_TIMEOUT_SECONDS", "28800"))
    print(f"Polling LLM managed job: {job_name}")

    interval, elapsed = 60, 0
    try:
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            try:
                all_jobs = _queue_managed_jobs(
                    refresh=(elapsed == interval),
                    skip_finished=False,
                    all_users=True,
                )
            except Exception as exc:
                print(f"[{elapsed}s] sky.jobs.queue(version=2) error: {exc} — retrying...")
                continue

            our = [j for j in all_jobs if j.get("job_name") == job_name]
            if not our:
                print(f"[{elapsed}s] Job '{job_name}' not visible yet")
                continue

            status = _status_text(our[-1].get("status", ""))
            print(f"[{elapsed}s] Status: {status}")
            if "SUCCEEDED" in status:
                print("LLM training completed successfully.")
                return
            if "FAILED" in status or "CANCELLED" in status:
                raise RuntimeError(f"LLM job '{job_name}' ended: {status}")

        raise RuntimeError(f"LLM job '{job_name}' timed out after {timeout}s.")

    except Exception:
        print(f"Cancelling LLM job '{job_name}'")
        try:
            import sky
            sky.get(sky.jobs.cancel(name=job_name))
        except Exception as exc:
            print(f"Warning: cancel failed: {exc}")
        raise


def cleanup_failed_training_controller():
    """Best-effort cleanup for failed training DAG runs.

    Strategy:
      1) Cancel current managed job by name (if provided).
      2) Inspect queue across all users.
      3) Tear down jobs controller only if there are no active managed jobs.
    """
    import importlib
    import sky
    controller_utils = importlib.import_module("sky.utils.controller_utils")

    job_name_raw = _env("SKY_JOB_NAME", "").strip()
    job_name = "" if job_name_raw in ("", "None", "null") else job_name_raw

    if job_name:
        print(f"Cleanup: cancelling managed job '{job_name}' (best effort)")
        try:
            sky.get(sky.jobs.cancel(name=job_name, all_users=True))
        except Exception as exc:
            print(f"[warn] cancel by name failed: {exc}")

    try:
        jobs = _queue_managed_jobs(refresh=True, skip_finished=False, all_users=True)
    except Exception as exc:
        print(f"[warn] could not query managed jobs before controller cleanup: {exc}")
        jobs = []

    active_jobs = [j for j in jobs if not _is_terminal_status(j.get("status"))]
    if active_jobs:
        sample = ", ".join(str(j.get("job_name", "<unnamed>")) for j in active_jobs[:3])
        print(
            f"Skipping jobs-controller teardown: {len(active_jobs)} managed jobs still active "
            f"({sample})."
        )
        return

    prefix = controller_utils.common.JOB_CONTROLLER_PREFIX
    try:
        clusters = sky.get(sky.status(all_users=True, refresh=False))
    except Exception as exc:
        print(f"[warn] could not list clusters for controller teardown: {exc}")
        return

    controller_names: list[str] = []
    for cluster in clusters or []:
        if isinstance(cluster, dict):
            name = str(cluster.get("name", "") or "")
        else:
            name = str(getattr(cluster, "name", "") or "")
        if name.startswith(prefix):
            controller_names.append(name)

    if not controller_names:
        print("No active SkyPilot jobs controller found to delete.")
        return

    for controller_name in controller_names:
        try:
            print(f"Tearing down SkyPilot jobs controller: {controller_name}")
            sky.get(sky.down(controller_name, purge=False))
            print(f"Controller deleted: {controller_name}")
        except Exception as exc:
            print(f"[warn] controller teardown failed for {controller_name}: {exc}")


def launch_vllm():
    import sky

    serve_run_id  = _env("SERVE_RUN_ID")
    model_id      = _env("HF_MODEL_ID")
    hf_token      = _env("HF_TOKEN")
    adapter_s3    = _env("LLM_ADAPTER_S3")
    vllm_port     = int(_env("VLLM_PORT", "8000"))
    max_model_len = int(_env("MAX_MODEL_LEN", "4096"))
    rc            = _rc()

    cluster_name = _k8s_name(serve_run_id, "vllm", max_len=32)
    yaml_path    = _yaml_path("vllm", rc)
    print(f"Using vLLM YAML: {yaml_path}")

    task = _load_task(yaml_path, serve_run_id, rc, prefer_spot=False)
    task.update_envs({
        "SERVE_RUN_ID":   serve_run_id,
        "HF_MODEL_ID":    model_id,
        "HF_TOKEN":       hf_token,
        "LLM_ADAPTER_S3": adapter_s3,
        "VLLM_PORT":      str(vllm_port),
        "MAX_MODEL_LEN":  str(max_model_len),
    })

    # detach_run=True: API server provisions cluster and starts vllm serve in
    # background. stream_and_get() returns once provisioning is done (not when
    # vllm serve exits — it never exits for a persistent serving cluster).
    sky.stream_and_get(
        sky.launch(task, cluster_name=cluster_name, retry_until_up=True, detach_run=True)
    )
    print(f"vLLM cluster launched: {cluster_name}")

    head_ip = _get_head_ip(cluster_name)
    print(f"Head IP: {head_ip or '(not yet available)'}")
    _xcom_push({"cluster_name": cluster_name, "head_ip": head_ip or "", "vllm_port": vllm_port})


def wait_vllm():
    import requests as _req

    info         = json.loads(_env("CLUSTER_INFO_JSON") or "{}")
    cluster_name = info.get("cluster_name", "")
    head_ip      = info.get("head_ip", "")
    vllm_port    = int(info.get("vllm_port") or _env("VLLM_PORT", "8000"))
    timeout      = int(_env("VLLM_HEALTH_TIMEOUT_SECONDS", "600"))
    interval     = 15

    if not head_ip and cluster_name:
        head_ip = _get_head_ip(cluster_name) or ""
    if not head_ip:
        raise RuntimeError(f"Cannot determine head IP for cluster '{cluster_name}'.")

    endpoint   = f"http://{head_ip}:{vllm_port}"
    health_url = f"{endpoint}/health"
    print(f"Polling vLLM health: {health_url} (timeout={timeout}s)")

    elapsed = 0
    while elapsed < timeout:
        try:
            r = _req.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"vLLM endpoint healthy after {elapsed}s: {endpoint}")
                _xcom_push(endpoint)
                return
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
        print(f"[{elapsed}s] vLLM not ready yet...")

    raise RuntimeError(f"vLLM endpoint '{health_url}' not healthy after {timeout}s.")


def launch_ray_vllm():
    """Multi-node Ray+vLLM cluster (Kimi-K2 pattern)."""
    import sky
    import yaml as _yaml

    serve_run_id      = _env("SERVE_RUN_ID")
    model_id          = _env("HF_MODEL_ID")
    hf_token          = _env("HF_TOKEN")
    adapter_s3        = _env("LLM_ADAPTER_S3")
    vllm_port         = int(_env("VLLM_PORT", "8081"))
    max_model_len     = int(_env("MAX_MODEL_LEN", "32768"))
    tensor_parallel   = int(_env("TENSOR_PARALLEL_SIZE", "8"))
    pipeline_parallel = int(_env("PIPELINE_PARALLEL_SIZE", "2"))
    rc                = _rc()

    cluster_name = _k8s_name(serve_run_id, "rvllm", max_len=32)
    yaml_path    = _yaml_path("vllm_multi", rc)
    print(f"Using Ray+vLLM YAML: {yaml_path}  ({pipeline_parallel} nodes)")

    # Load base YAML and override num_nodes from pipeline_parallel_size
    base_task = _load_task(yaml_path, serve_run_id, rc, prefer_spot=False)

    # Re-open the (possibly rewritten) YAML to set num_nodes
    tmp_src = f"/tmp/sky_ray_vllm_base_{serve_run_id}.yaml"
    with open(yaml_path) as fh:
        sky_conf = _yaml.safe_load(fh)
    sky_conf["num_nodes"] = pipeline_parallel
    with open(tmp_src, "w") as fh:
        _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)
    task = sky.Task.from_yaml(tmp_src)

    task.update_envs({
        "SERVE_RUN_ID":           serve_run_id,
        "HF_MODEL_ID":            model_id,
        "HF_TOKEN":               hf_token,
        "LLM_ADAPTER_S3":         adapter_s3,
        "VLLM_PORT":              str(vllm_port),
        "MAX_MODEL_LEN":          str(max_model_len),
        "TENSOR_PARALLEL_SIZE":   str(tensor_parallel),
        "PIPELINE_PARALLEL_SIZE": str(pipeline_parallel),
    })

    sky.stream_and_get(
        sky.launch(task, cluster_name=cluster_name, retry_until_up=True, detach_run=True)
    )
    print(f"Ray+vLLM cluster launched: {cluster_name} ({pipeline_parallel} nodes)")

    head_ip = _get_head_ip(cluster_name)
    print(f"Head IP: {head_ip or '(not yet available)'}")
    _xcom_push({"cluster_name": cluster_name, "head_ip": head_ip or "", "vllm_port": vllm_port})


def wait_ray_vllm():
    """Poll /health for the multi-node Ray+vLLM cluster (longer startup)."""
    import requests as _req

    info         = json.loads(_env("CLUSTER_INFO_JSON") or "{}")
    cluster_name = info.get("cluster_name", "")
    head_ip      = info.get("head_ip", "")
    vllm_port    = int(info.get("vllm_port") or _env("VLLM_PORT", "8081"))
    timeout      = int(_env("VLLM_HEALTH_TIMEOUT_SECONDS", "900"))
    interval     = 20   # multi-node startup takes longer

    if not head_ip and cluster_name:
        head_ip = _get_head_ip(cluster_name) or ""
    if not head_ip:
        raise RuntimeError(f"Cannot determine head IP for cluster '{cluster_name}'.")

    endpoint   = f"http://{head_ip}:{vllm_port}"
    health_url = f"{endpoint}/health"
    print(f"Polling Ray+vLLM health: {health_url} (timeout={timeout}s)")

    elapsed = 0
    while elapsed < timeout:
        try:
            r = _req.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"Ray+vLLM endpoint healthy after {elapsed}s: {endpoint}")
                _xcom_push(endpoint)
                return
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
        print(f"[{elapsed}s] Ray+vLLM not ready yet...")

    raise RuntimeError(f"Ray+vLLM endpoint '{health_url}' not healthy after {timeout}s.")


def run_training():
    """Single-pod full training lifecycle: submit → poll → cleanup on failure.

    Runs all SkyPilot calls within one pod so a single local API server handles
    the entire job lifecycle, eliminating cold-start overhead and cross-pod
    state fragmentation.
    """
    train_run_id      = _env("TRAIN_RUN_ID")
    preprocess_run_id = _env("PREPROCESS_RUN_ID")
    processed_table   = _env("PROCESSED_TABLE")
    model_type        = _env("MODEL_TYPE", "xgboost")
    params_s3_path    = _env("PARAMS_S3_PATH")
    num_nodes         = int(_env("NUM_NODES", "1"))
    mlflow_uri        = _env("MLFLOW_TRACKING_URI")
    rc                = _rc()
    timeout           = int(_env("SKY_TIMEOUT_SECONDS", "7200"))

    job_name = _k8s_name(train_run_id, "sky")
    use_gpu  = model_type in ("pytorch", "ssm", "bae")
    kind     = "train_multi" if (use_gpu and num_nodes > 1) else "train"

    yaml_path = _yaml_path(kind, rc)
    print(f"[run-training] YAML: {yaml_path}  (model={model_type} gpu={use_gpu} nodes={num_nodes})")

    task = _load_task(yaml_path, train_run_id, rc if use_gpu else None)
    envs: dict[str, str] = {
        "TRAIN_RUN_ID":        train_run_id,
        "PREPROCESS_RUN_ID":   preprocess_run_id,
        "MODEL_TYPE":          model_type,
        "PARAMS_S3_PATH":      params_s3_path,
        "PROCESSED_TABLE":     processed_table,
        "MLFLOW_TRACKING_URI": mlflow_uri,
    }
    if num_nodes > 1:
        envs["NUM_WORKERS"] = str(num_nodes * 8)
    task.update_envs(envs)

    # ── Submit ────────────────────────────────────────────────────────────────
    _launch_managed_job_with_stream_fallback(task, job_name)
    print(f"[run-training] Managed job launched: {job_name}")
    _xcom_push(job_name)

    # ── Poll (same API server — no state fragmentation) ───────────────────────
    print(f"[run-training] Polling managed job: {job_name} (timeout={timeout}s)")
    interval, elapsed = 30, 0
    try:
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            try:
                all_jobs = _queue_managed_jobs(
                    refresh=(elapsed == interval),
                    skip_finished=False,
                    all_users=True,
                )
            except Exception as exc:
                print(f"[{elapsed}s] sky.jobs.queue error: {exc} — retrying...")
                continue

            our = [j for j in all_jobs if j.get("job_name") == job_name]
            if not our:
                print(f"[{elapsed}s] Job '{job_name}' not visible yet (provisioning...)")
                continue

            status = _status_text(our[-1].get("status", ""))
            print(f"[{elapsed}s] Status: {status}")
            if "SUCCEEDED" in status:
                print("[run-training] Training completed successfully.")
                return
            if "FAILED" in status or "CANCELLED" in status:
                raise RuntimeError(f"Job '{job_name}' ended with status: {status}")

        raise RuntimeError(f"Job '{job_name}' timed out after {timeout}s.")

    except Exception:
        print(f"[run-training] Cancelling job '{job_name}' due to error/timeout.")
        try:
            import sky
            sky.get(sky.jobs.cancel(name=job_name))
        except Exception as exc:
            print(f"Warning: cancel failed: {exc}")
        raise


def run_llm():
    """Single-pod full LLM training lifecycle: submit → poll → cancel on failure.

    Runs all SkyPilot calls within one pod so a single local API server handles
    the entire job lifecycle, eliminating cold-start overhead and cross-pod
    state fragmentation.
    """
    run_id     = _env("LLM_TRAIN_RUN_ID")
    model_id   = _env("LLM_MODEL_ID")
    dataset_s3 = _env("TRAIN_DATASET_S3")
    hf_token   = _env("HF_TOKEN")
    max_steps  = int(_env("MAX_STEPS", "500"))
    save_steps = int(_env("SAVE_STEPS", "100"))
    lora_on    = _env("LORA_ENABLED", "true")
    lora_rank  = int(_env("LORA_RANK", "8"))
    ds_stage   = int(_env("DEEPSPEED_STAGE", "3"))
    mlflow_uri = _env("MLFLOW_TRACKING_URI")
    rc         = _rc()
    timeout    = int(_env("SKY_TIMEOUT_SECONDS", "28800"))

    job_name  = _k8s_name(run_id, "llm")
    yaml_path = _yaml_path("llm", rc)
    print(f"[run-llm] LLM YAML: {yaml_path}")

    task = _load_task(yaml_path, run_id, rc)
    task.update_envs({
        "LLM_TRAIN_RUN_ID":    run_id,
        "LLM_MODEL_ID":        model_id,
        "HF_TOKEN":            hf_token,
        "TRAIN_DATASET_S3":    dataset_s3,
        "MAX_STEPS":           str(max_steps),
        "SAVE_STEPS":          str(save_steps),
        "LORA_ENABLED":        lora_on,
        "LORA_RANK":           str(lora_rank),
        "DEEPSPEED_STAGE":     str(ds_stage),
        "MLFLOW_TRACKING_URI": mlflow_uri,
    })

    # ── Submit ────────────────────────────────────────────────────────────────
    _launch_managed_job_with_stream_fallback(task, job_name)
    print(f"[run-llm] LLM managed job launched: {job_name}")
    _xcom_push(job_name)

    # ── Poll (same API server — no state fragmentation) ───────────────────────
    print(f"[run-llm] Polling LLM managed job: {job_name} (timeout={timeout}s)")
    interval, elapsed = 60, 0
    try:
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            try:
                all_jobs = _queue_managed_jobs(
                    refresh=(elapsed == interval),
                    skip_finished=False,
                    all_users=True,
                )
            except Exception as exc:
                print(f"[{elapsed}s] sky.jobs.queue error: {exc} — retrying...")
                continue

            our = [j for j in all_jobs if j.get("job_name") == job_name]
            if not our:
                print(f"[{elapsed}s] Job '{job_name}' not visible yet")
                continue

            status = _status_text(our[-1].get("status", ""))
            print(f"[{elapsed}s] Status: {status}")
            if "SUCCEEDED" in status:
                print("[run-llm] LLM training completed successfully.")
                return
            if "FAILED" in status or "CANCELLED" in status:
                raise RuntimeError(f"LLM job '{job_name}' ended: {status}")

        raise RuntimeError(f"LLM job '{job_name}' timed out after {timeout}s.")

    except Exception:
        print(f"[run-llm] Cancelling LLM job '{job_name}'")
        try:
            import sky
            sky.get(sky.jobs.cancel(name=job_name))
        except Exception as exc:
            print(f"Warning: cancel failed: {exc}")
        raise


def register_endpoint():
    import boto3
    from datetime import datetime

    endpoint_url = _env("ENDPOINT_URL")
    serve_run_id = _env("SERVE_RUN_ID")
    model_id     = _env("HF_MODEL_ID")
    cluster_name = _env("CLUSTER_NAME", "")
    orchestration = _env("ORCHESTRATION", "vllm_single_node")
    bucket       = _env("S3_BUCKET", "k8s-mlops-platform-bucket")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    key = f"runs/serving/{serve_run_id}/endpoint.json"
    payload: dict = {
        "endpoint_url":  endpoint_url,
        "model_id":      model_id,
        "serve_run_id":  serve_run_id,
        "orchestration": orchestration,
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }
    if cluster_name:
        payload["cluster_name"] = cluster_name
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode(),
        ContentType="application/json",
    )
    print(f"Endpoint registered: {endpoint_url} → s3://{bucket}/{key}")


# ─── Entry point ──────────────────────────────────────────────────────────────

_COMMANDS = {
    # ── Unified single-pod lifecycle (preferred) ──────────────────────────────
    "run-training":      run_training,
    "run-llm":           run_llm,
    # ── Legacy split-phase commands (kept for reference/debugging) ────────────
    "submit-training":   submit_training,
    "poll-training":     poll_training,
    "cleanup-failed-training-controller": cleanup_failed_training_controller,
    "submit-llm":        submit_llm,
    "poll-llm":          poll_llm,
    # ── vLLM serving (persistent cluster — split-phase is intentional here) ───
    "launch-vllm":       launch_vllm,
    "wait-vllm":         wait_vllm,
    "launch-ray-vllm":   launch_ray_vllm,
    "wait-ray-vllm":     wait_ray_vllm,
    "register-endpoint": register_endpoint,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        cmds = " | ".join(_COMMANDS)
        print(f"Usage: sky_runner.py [{cmds}]", file=sys.stderr)
        sys.exit(1)
    _COMMANDS[sys.argv[1]]()
