#!/usr/bin/env python3
"""SkyPilot Runner — CLI script for Airflow KubernetesPodOperator.

Runs inside takenking9879/sky-runner:0.12.0.
All inputs come from environment variables; return values are written to
`SKY_RUNNER_XCOM_PATH` (default `/airflow/xcom/return.json`) so
KubernetesPodOperator do_xcom_push=True works.

Commands:
    submit-training     Launch managed tabular training job; pushes job_name (str)
    poll-training       Poll managed training job until terminal state
    cleanup-failed-training-controller
                        Best-effort cleanup: cancel failed training job and
                        tear down SkyPilot jobs controller when safe
    submit-llm          Launch managed LLM fine-tuning job; pushes job_name (str)
    poll-llm            Poll managed LLM job until terminal state
    launch-vllm             Provision persistent single-node vLLM cluster; pushes cluster_info (dict)
    wait-vllm               Poll /health until ready; pushes endpoint_url (str)
    launch-ray-vllm         Provision persistent multi-node Ray+vLLM cluster; pushes cluster_info (dict)
    wait-ray-vllm           Poll /health until ready (multi-node); pushes endpoint_url (str)
    launch-tabular-serve    Deploy sky serve service for tabular model; pushes service_info (dict)
    wait-tabular-serve      Poll sky.serve.status() until READY; pushes endpoint_url (str)
    register-endpoint       Write endpoint URL to S3 (shared by all serving types)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _enable_line_buffered_output() -> None:
    """Best-effort unbuffered-ish output for Airflow task logs."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


_enable_line_buffered_output()

# ─── XCom ─────────────────────────────────────────────────────────────────────

_DEFAULT_XCOM_PATH = Path("/airflow/xcom/return.json")


def _xcom_output_path() -> Path:
    custom_path = os.getenv("SKY_RUNNER_XCOM_PATH", "").strip()
    return Path(custom_path) if custom_path else _DEFAULT_XCOM_PATH


def _xcom_push(value) -> None:
    xcom_path = _xcom_output_path()
    xcom_path.parent.mkdir(parents=True, exist_ok=True)
    xcom_path.write_text(json.dumps(value))


# Hardcoded SkyServe wait log-streaming defaults (live mode only).
_SERVE_LOG_MAX_REPLICAS = 2
_SERVE_STREAM_CONTROLLER = False
_SERVE_STREAM_LOAD_BALANCER = False
_SERVE_FALLBACK_REPLICA_IDS = ("1",)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SKY_CLI_PREFIX: list[str] | None = None


def _sky_cli_cmd(*args: str) -> list[str]:
    """Build a Sky CLI command robustly for containerized Airflow tasks.

    Prefer an absolute sky binary path to avoid PATH collisions (e.g. k3s/sky
    directory shadowing) and preserve CLI follow semantics.
    """
    global _SKY_CLI_PREFIX
    if _SKY_CLI_PREFIX is None:
        venv_sky = Path(sys.executable).with_name("sky")
        if venv_sky.exists() and os.access(venv_sky, os.X_OK):
            _SKY_CLI_PREFIX = [str(venv_sky)]
        else:
            discovered = shutil.which("sky")
            if discovered:
                _SKY_CLI_PREFIX = [discovered]
            else:
                _SKY_CLI_PREFIX = [sys.executable, "-m", "sky.cli"]
                print(
                    "[serve-live][warn] Could not resolve executable sky binary; "
                    "falling back to python -m sky.cli."
                )

        print(f"[serve-live] sky cli invocation: {' '.join(_SKY_CLI_PREFIX)}")

    return [*_SKY_CLI_PREFIX, *args]


def _run_cmd_capture(cmd: list[str], timeout_s: int = 25) -> tuple[int, str]:
    """Run command and capture combined stdout/stderr safely.

    Returns (exit_code, output). Timeout returns code 124 with partial output.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        output = ""
        if exc.stdout:
            output += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
        if exc.stderr:
            output += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
        return 124, output
    except OSError as exc:
        return 126, f"process launch failed ({exc})"
    except Exception as exc:
        return 1, f"command execution failed ({exc})"


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _discover_serve_replica_ids(service_name: str, max_replicas: int) -> tuple[list[str], str]:
    # Parse IDs from the "Service Replicas" table:
    # <service_name>  <id>  <version> ...
    cmd = _sky_cli_cmd("serve", "status", service_name)
    rc, output = _run_cmd_capture(cmd, timeout_s=20)
    if rc != 0:
        snippet_lines = [ln.strip() for ln in output.splitlines() if ln.strip()][:2]
        snippet = " | ".join(snippet_lines) if snippet_lines else "<no output>"
        return [], f"status rc={rc}, snippet={snippet[:220]}"

    ids: list[str] = []
    strict_pattern = re.compile(rf"\b{re.escape(service_name)}\b\s+(\d+)\b")
    for line in output.splitlines():
        clean_line = _strip_ansi(line).strip()
        if not clean_line or service_name not in clean_line:
            continue

        rid: str | None = None
        m = strict_pattern.search(clean_line)
        if m:
            rid = m.group(1)
        else:
            parts = re.split(r"\s+", clean_line)
            try:
                start_idx = parts.index(service_name)
            except ValueError:
                start_idx = -1
            if start_idx >= 0:
                for token in parts[start_idx + 1 :]:
                    if token.isdigit():
                        rid = token
                        break

        if not rid:
            continue
        if rid not in ids:
            ids.append(rid)

    if not ids:
        return [], "status ok but no replica IDs parsed yet"

    return ids[: max(1, max_replicas)], ""

class _SkyServeLiveStreamer:
    """Streams SkyServe logs in follow mode (replicas by default)."""

    def __init__(
        self,
        service_name: str,
        max_replicas: int = 2,
        include_controller: bool = False,
        include_load_balancer: bool = True,
    ):
        self.service_name = service_name
        self.max_replicas = max(1, max_replicas)
        self.include_controller = include_controller
        self.include_load_balancer = include_load_balancer
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._warned_once: set[str] = set()
        self._primary_discovery_attempts = 0
        self._primary_discovery_successes = 0
        self._fallback_uses = 0
        self._fallback_active = False

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        print(msg)

    def _pump_output(self, label: str, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    print(f"[serve-live][{label}] {line}")
        finally:
            rc = proc.poll()
            if rc is None:
                try:
                    rc = proc.wait(timeout=1)
                except Exception:
                    rc = None
            print(f"[serve-live][{label}] stream ended (rc={rc if rc is not None else 'unknown'})")

    def _start_follow(self, label: str, cmd: list[str]) -> None:
        if label in self._procs:
            proc = self._procs[label]
            if proc.poll() is None:
                return

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._warn_once(
                f"start:{label}",
                f"[serve-live][{label}] could not start streaming ({exc}); cmd={' '.join(cmd)}",
            )
            return

        self._procs[label] = proc
        thread = threading.Thread(target=self._pump_output, args=(label, proc), daemon=True)
        thread.start()
        print(f"[serve-live][{label}] streaming started")

    def start_base_streams(self) -> None:
        if self.include_controller:
            self._start_follow(
                "controller",
                _sky_cli_cmd("serve", "logs", "--follow", "--controller", self.service_name),
            )
        if self.include_load_balancer:
            self._start_follow(
                "load-balancer",
                _sky_cli_cmd("serve", "logs", "--follow", "--load-balancer", self.service_name),
            )

    def refresh_replica_streams(self) -> None:
        self._primary_discovery_attempts += 1
        replica_ids, primary_reason = _discover_serve_replica_ids(
            self.service_name, self.max_replicas
        )
        if replica_ids:
            self._primary_discovery_successes += 1
            if self._fallback_active:
                print(
                    "[serve-live] primary replica discovery recovered; "
                    f"using discovered IDs: {','.join(replica_ids)}"
                )
                self._fallback_active = False

        if not replica_ids:
            self._warn_once(
                "replica-primary-empty",
                "[serve-live] primary replica discovery returned no IDs "
                f"({primary_reason})",
            )
            replica_ids = list(_SERVE_FALLBACK_REPLICA_IDS)
            self._fallback_uses += 1
            if self._fallback_uses == 1:
                print(
                    "[serve-live] fallback triggered: using static replica IDs "
                    f"{','.join(replica_ids)}"
                )
            elif self._fallback_uses % 5 == 0:
                print(
                    "[serve-live] fallback still active "
                    f"(uses={self._fallback_uses}, primary_successes={self._primary_discovery_successes})"
                )
            self._fallback_active = True

        for replica_id in replica_ids:
            label = f"replica-{replica_id}"
            self._start_follow(
                label,
                _sky_cli_cmd(
                    "serve",
                    "logs",
                    "--follow",
                    self.service_name,
                    replica_id,
                ),
            )

    def stop(self) -> None:
        for label, proc in list(self._procs.items()):
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception as exc:
                self._warn_once(f"stop-term:{label}", f"[serve-live][{label}] terminate failed: {exc}")

        deadline = time.time() + 3
        for label, proc in list(self._procs.items()):
            if proc.poll() is not None:
                continue
            try:
                remaining = max(0.1, deadline - time.time())
                proc.wait(timeout=remaining)
            except Exception:
                try:
                    proc.kill()
                except Exception as exc:
                    self._warn_once(f"stop-kill:{label}", f"[serve-live][{label}] kill failed: {exc}")

        print(
            "[serve-live] replica discovery summary: "
            f"primary_attempts={self._primary_discovery_attempts}, "
            f"primary_successes={self._primary_discovery_successes}, "
            f"fallback_uses={self._fallback_uses}"
        )
        if self._fallback_uses > 0 and self._primary_discovery_successes == 0:
            print(
                "[serve-live][warn] fallback was used but primary discovery never succeeded; "
                "fallback acted as principal path and primary parsing should be revisited."
            )

# ─── Environment helpers ───────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _aws_runtime_envs() -> dict[str, str]:
    """Build AWS env vars to forward into remote SkyPilot tasks.

    Keeps credentials out of logs while still surfacing enough context to debug
    S3 connectivity issues from remote replicas.
    """
    access_key = _env("AWS_ACCESS_KEY_ID").strip()
    secret_key = _env("AWS_SECRET_ACCESS_KEY").strip()
    session_token = _env("AWS_SESSION_TOKEN").strip()
    region = (
        _env("AWS_REGION").strip()
        or _env("AWS_DEFAULT_REGION").strip()
        or "us-east-1"
    )
    endpoint = (
        _env("AWS_S3_ENDPOINT_URL").strip()
        or _env("AWS_ENDPOINT_URL").strip()
    )

    envs: dict[str, str] = {
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
    }
    if access_key:
        envs["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        envs["AWS_SECRET_ACCESS_KEY"] = secret_key
    if session_token:
        envs["AWS_SESSION_TOKEN"] = session_token
    if endpoint:
        envs["AWS_S3_ENDPOINT_URL"] = endpoint
        envs["AWS_ENDPOINT_URL"] = endpoint

    has_creds = bool(access_key and secret_key)
    print(
        "[aws-forward]"
        f" region={region}"
        f" endpoint={'set' if endpoint else 'default'}"
        f" creds={'yes' if has_creds else 'no'}"
        f" session_token={'yes' if bool(session_token) else 'no'}"
    )
    if not has_creds:
        print(
            "[warn] AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY not set for SkyServe task; "
            "private S3 downloads in remote replicas may fail."
        )

    return envs


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


def _detect_k3s_gpu_accelerator_kubectl(n_gpus: int = 1, label: str = "") -> str:
    """Return 'accelerator:n_gpus'. Reads K8S_GPU_ACCELERATOR env var set by Airflow DAG."""
    prefix = f"[{label}] " if label else ""
    acc = os.environ.get("K8S_GPU_ACCELERATOR", "").strip()
    if acc:
        print(f"{prefix}K3S GPU from env K8S_GPU_ACCELERATOR: {acc}")
        return acc
    return ""


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge src into dst and return dst."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _extract_serve_controller_cfg() -> dict | None:
    """Read SKYSERVE_CONTROLLER_CONFIG_JSON and normalize it to controller-only shape."""
    raw = _env("SKYSERVE_CONTROLLER_CONFIG_JSON")
    if not raw or raw in ("null", "{}"):
        return None

    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(f"[warn] Invalid SKYSERVE_CONTROLLER_CONFIG_JSON ({exc}); skipping")
        return None

    if not isinstance(payload, dict):
        print("[warn] SKYSERVE_CONTROLLER_CONFIG_JSON must be a JSON object; skipping")
        return None

    # Accept one of:
    #  1) {"high_availability": ..., "resources": {...}}
    #  2) {"controller": {...}}
    #  3) {"serve": {"controller": {...}}}
    if isinstance(payload.get("serve"), dict):
        payload = payload["serve"].get("controller", {})
    elif isinstance(payload.get("controller"), dict):
        payload = payload["controller"]

    if not isinstance(payload, dict) or not payload:
        return None

    normalized: dict = {}
    if "high_availability" in payload and isinstance(payload["high_availability"], bool):
        normalized["high_availability"] = payload["high_availability"]

    resources = payload.get("resources")
    if isinstance(resources, dict):
        resources_norm: dict = {}
        for key in ("infra", "cpus", "disk_size"):
            value = resources.get(key)
            if value is not None and value != "":
                resources_norm[key] = value
        if resources_norm:
            normalized["resources"] = resources_norm

    return normalized or None


def _apply_serve_controller_config() -> None:
    """Merge optional serve.controller settings into ~/.sky/config.yaml.

    SkyServe controller tuning is consumed from sky config, not from task YAML.
    """
    controller_cfg = _extract_serve_controller_cfg()
    if not controller_cfg:
        return

    import yaml as _yaml

    config_path = Path.home() / ".sky" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_data: dict = {}
    if config_path.exists():
        try:
            with open(config_path) as fh:
                loaded = _yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    config_data = loaded
        except Exception as exc:
            print(f"[warn] Failed to read existing {config_path}: {exc}; rebuilding file")

    serve_block = config_data.setdefault("serve", {})
    if not isinstance(serve_block, dict):
        config_data["serve"] = {}
        serve_block = config_data["serve"]

    controller_block = serve_block.setdefault("controller", {})
    if not isinstance(controller_block, dict):
        serve_block["controller"] = {}
        controller_block = serve_block["controller"]

    _deep_merge(controller_block, controller_cfg)

    with open(config_path, "w") as fh:
        _yaml.safe_dump(config_data, fh, default_flow_style=False, sort_keys=False)

    print(f"Applied SkyServe controller config to {config_path}: {controller_cfg}")


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
    ("vllm",             "runpod"): "vllm-serving-runpod.yaml",
    ("vllm",             "vast"):   "vllm-serving-vast.yaml",
    ("vllm_multi",       "aws"):    "ray-vllm-multinode-serving.yaml",
    ("tabular_serve",    "runpod"):   "tabular-serving-single.yaml",
    ("tabular_serve",    "vast"):     "tabular-serving-single.yaml",
    ("tabular_serve",    "aws"):      "tabular-serving-single.yaml",
    ("tabular_serve",    "localgpu"): "tabular-serving-k8s.yaml",
    ("tabular_serve",    "k8s"):      "tabular-serving-k8s.yaml",
    ("tabular_serve_multi", "runpod"): "tabular-serving-multinode.yaml",
    ("tabular_serve_multi", "vast"):   "tabular-serving-multinode.yaml",
    ("tabular_serve_multi", "aws"):    "tabular-serving-multinode.yaml",
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
    """Load a SkyPilot Task from yaml_path, optionally replacing resources with an
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
            resources_cfg = sky_conf.setdefault("resources", {})
            resources_cfg["ordered"] = gpu_fallbacks
            resources_cfg.pop("any_of", None)
            resources_cfg.pop("infra", None)
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
        resources_cfg = sky_conf.setdefault("resources", {})
        resources_cfg["ordered"] = result.any_of
        resources_cfg.pop("any_of", None)
        resources_cfg.pop("infra", None)

        tmp = f"/tmp/sky_task_{run_id}.yaml"
        with open(tmp, "w") as fh:
            _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

        print(
            f"Dynamic ordered: {len(result.any_of)} entries "
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
    if not slug:
        slug = prefix
    if len(slug) <= max_len:
        return slug

    # Preserve deterministic uniqueness when truncation is required.
    suffix = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    base_max_len = max(1, max_len - len(suffix) - 1)
    trimmed = slug[:base_max_len].rstrip("-") or slug[:base_max_len]
    return f"{trimmed}-{suffix}"


def _managed_job_name(name: str, prefix: str, max_len: int = 40) -> str:
    """Build a managed-job name unique to this Airflow task attempt."""
    base = _k8s_name(name, prefix, max_len=max_len) or prefix
    seed = (
        f"{os.getenv('AIRFLOW_CTX_DAG_RUN_ID', '')}-"
        f"{os.getenv('AIRFLOW_CTX_TASK_ID', '')}-"
        f"{os.getenv('AIRFLOW_CTX_TRY_NUMBER', '')}-"
        f"{time.time_ns()}-{os.getpid()}"
    )
    suffix = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]

    base_max_len = max(1, max_len - len(suffix) - 1)
    trimmed = base[:base_max_len].rstrip("-") or prefix
    return f"{trimmed}-{suffix}"


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


def _failure_details(job: dict) -> str:
    """Build a compact diagnostics string from queue(version=2) job fields."""
    fields = [
        "failure_reason",
        "failure_type",
        "message",
        "cluster_name",
        "job_id",
        "task_id",
    ]
    details: list[str] = []
    for key in fields:
        val = job.get(key)
        if val not in (None, "", [], {}):
            details.append(f"{key}={val}")
    return "; ".join(details)


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

    job_name = _managed_job_name(train_run_id, "sky")
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
                details = _failure_details(our[-1])
                if details:
                    raise RuntimeError(
                        f"Job '{job_name}' ended with status: {status}. Details: {details}"
                    )
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

    job_name  = _managed_job_name(run_id, "llm")
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
                details = _failure_details(our[-1])
                if details:
                    raise RuntimeError(
                        f"LLM job '{job_name}' ended: {status}. Details: {details}"
                    )
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
    task_envs = {
        "SERVE_RUN_ID":   serve_run_id,
        "HF_MODEL_ID":    model_id,
        "HF_TOKEN":       hf_token,
        "LLM_ADAPTER_S3": adapter_s3,
        "VLLM_PORT":      str(vllm_port),
        "MAX_MODEL_LEN":  str(max_model_len),
    }
    task_envs.update(_aws_runtime_envs())
    task.update_envs(task_envs)

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
    """Multi-node Ray+vLLM serving via SkyServe (Kimi-K2 pattern)."""
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
    replicas          = int(_env("REPLICAS", "1"))
    rc                = _rc()

    service_name = _k8s_name(serve_run_id, "rvllm", max_len=32)
    yaml_path    = _yaml_path("vllm_multi", rc)
    print(f"Using Ray+vLLM YAML: {yaml_path}  ({pipeline_parallel} nodes × {replicas} replicas)")

    _apply_serve_controller_config()

    # Load base YAML and override num_nodes + service.replicas
    tmp_src = f"/tmp/sky_ray_vllm_base_{serve_run_id}.yaml"
    with open(yaml_path) as fh:
        sky_conf = _yaml.safe_load(fh)
    sky_conf["num_nodes"] = pipeline_parallel
    service_cfg = sky_conf.setdefault("service", {})
    service_cfg["replicas"] = replicas
    service_cfg["ports"] = int(vllm_port)
    with open(tmp_src, "w") as fh:
        _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)
    task = sky.Task.from_yaml(tmp_src)

    task_envs = {
        "SERVE_RUN_ID":           serve_run_id,
        "HF_MODEL_ID":            model_id,
        "HF_TOKEN":               hf_token,
        "LLM_ADAPTER_S3":         adapter_s3,
        "VLLM_PORT":              str(vllm_port),
        "MAX_MODEL_LEN":          str(max_model_len),
        "TENSOR_PARALLEL_SIZE":   str(tensor_parallel),
        "PIPELINE_PARALLEL_SIZE": str(pipeline_parallel),
    }
    task_envs.update(_aws_runtime_envs())
    task.update_envs(task_envs)

    sky.stream_and_get(sky.serve.up(task, service_name=service_name))
    print(f"Ray+vLLM service launched: {service_name} ({pipeline_parallel} nodes × {replicas} replicas)")
    _xcom_push({"service_name": service_name, "vllm_port": vllm_port})


def wait_ray_vllm():
    """Poll sky.serve.status() until the Ray+vLLM service is READY."""
    import sky

    info         = json.loads(_env("SERVICE_INFO_JSON") or "{}")
    service_name = info.get("service_name", "")
    timeout      = int(_env("VLLM_HEALTH_TIMEOUT_SECONDS", "900"))
    interval     = 30   # multi-node startup takes longer

    if not service_name:
        raise RuntimeError("SERVICE_INFO_JSON must contain 'service_name'")

    print(f"Polling sky serve status for Ray+vLLM service '{service_name}' (timeout={timeout}s)")
    live_streamer = _SkyServeLiveStreamer(
        service_name=service_name,
        max_replicas=_SERVE_LOG_MAX_REPLICAS,
        include_controller=_SERVE_STREAM_CONTROLLER,
        include_load_balancer=_SERVE_STREAM_LOAD_BALANCER,
    )
    live_streamer.start_base_streams()
    live_streamer.refresh_replica_streams()
    print(
        "[serve-live] streaming replica logs only "
        f"(follow_only=True, replicas={_SERVE_LOG_MAX_REPLICAS}, "
        f"controller={_SERVE_STREAM_CONTROLLER}, "
        f"load_balancer={_SERVE_STREAM_LOAD_BALANCER})"
    )

    elapsed = 0
    try:
        while elapsed < timeout:
            try:
                statuses = sky.get(sky.serve.status(service_names=[service_name]))
                if statuses:
                    svc = statuses[0]
                    if isinstance(svc, dict):
                        endpoint   = svc.get("endpoint_url") or svc.get("endpoint", "")
                        status_val = str(svc.get("status", "")).upper()
                    else:
                        endpoint   = getattr(svc, "endpoint_url", None) or getattr(svc, "endpoint", "")
                        status_val = str(getattr(svc, "status", "")).upper()
                    if "READY" in status_val and endpoint:
                        print(f"Ray+vLLM service healthy after {elapsed}s: {endpoint}")
                        _xcom_push(endpoint)
                        return
                    if "FAILED" in status_val:
                        raise RuntimeError(f"Service '{service_name}' entered FAILED state.")
                    print(f"[{elapsed}s] Status={status_val}, endpoint={endpoint or '—'}")
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"[{elapsed}s] status check error: {exc}")

            live_streamer.refresh_replica_streams()

            time.sleep(interval)
            elapsed += interval

        raise RuntimeError(
            f"Ray+vLLM service '{service_name}' not READY after {timeout}s."
        )
    finally:
        live_streamer.stop()


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
    use_deepspeed     = _env("USE_DEEPSPEED", "false")
    deepspeed_stage   = _env("DEEPSPEED_STAGE", "1")
    rc                = _rc()
    timeout           = int(_env("SKY_TIMEOUT_SECONDS", "7200"))

    job_name = _managed_job_name(train_run_id, "sky")
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
        "USE_DEEPSPEED":       use_deepspeed,
        "DEEPSPEED_STAGE":     deepspeed_stage,
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
                details = _failure_details(our[-1])
                if details:
                    raise RuntimeError(
                        f"Job '{job_name}' ended with status: {status}. Details: {details}"
                    )
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

    job_name  = _managed_job_name(run_id, "llm")
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
                details = _failure_details(our[-1])
                if details:
                    raise RuntimeError(
                        f"LLM job '{job_name}' ended: {status}. Details: {details}"
                    )
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


def launch_tabular_serve():
    """Deploy a SkyPilot sky serve service for tabular model inference.

    Uses sky.serve.up() (managed service) — NOT sky.launch() (persistent cluster).
    Single-node: tabular-serving-single.yaml
    Multi-node:  tabular-serving-multinode.yaml (Kimi-K2 pattern adapted for tabular)
    """
    import sky
    import yaml as _yaml

    serve_run_id   = _env("SERVE_RUN_ID")
    registry_model = _env("REGISTRY_MODEL_NAME")
    alias          = _env("MODEL_ALIAS", "champion")
    serve_port     = int(_env("SERVE_PORT", "8000"))
    min_replicas   = int(_env("MIN_REPLICAS", "1"))
    max_replicas   = int(_env("MAX_REPLICAS", "3"))
    target_qps     = int(_env("TARGET_QPS_PER_REPLICA", "10"))
    num_nodes      = int(_env("NUM_NODES", "1"))
    rc             = _rc()
    mlflow_tracking_uri = _env("MLFLOW_TRACKING_URI").strip()

    kind = "tabular_serve_multi" if num_nodes > 1 else "tabular_serve"
    yaml_path = _yaml_path(kind, rc)
    print(f"Using tabular serve YAML: {yaml_path}  (num_nodes={num_nodes})")

    _apply_serve_controller_config()

    with open(yaml_path) as fh:
        sky_conf = _yaml.safe_load(fh)

    service_cfg = sky_conf.setdefault("service", {})
    service_cfg["ports"] = int(serve_port)

    # Inject replica policy
    service_cfg.setdefault("replica_policy", {}).update({
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "target_qps_per_replica": target_qps,
    })
    if num_nodes > 1:
        sky_conf["num_nodes"] = num_nodes

    # Inject resource constraints
    provider = _provider(rc)
    if provider in ("localgpu", "k8s"):
        n_gpus = max(1, int((rc or {}).get("num_gpus_per_node") or 1))
        k8s_acc = _detect_k3s_gpu_accelerator_kubectl(n_gpus, label="tabular-serve")
        if not k8s_acc:
            raise RuntimeError(
                "[tabular-serve] localGPU selected but no K8s node with "
                "nvidia.com/gpu and skypilot.co/accelerator label found."
            )
        resources_cfg = sky_conf.setdefault("resources", {})
        resources_cfg["infra"] = "k8s/in-cluster"
        resources_cfg["accelerators"] = k8s_acc
        resources_cfg["image_id"] = "docker:takenking9879/ray-train:2.53.0"
        resources_cfg.pop("disk_size", None)  # not supported by K8s
        resources_cfg.pop("ordered", None)
        resources_cfg.pop("any_of", None)
        # Allow memory override via resource_constraints; default in YAML is "8+"
        mem_gb = (rc or {}).get("memory_gb_per_node")
        if mem_gb:
            resources_cfg["memory"] = f"{mem_gb}+"
        print(f"[tabular-serve] K3S GPU: {k8s_acc} — infra: k8s/in-cluster, memory={resources_cfg.get('memory', 'yaml-default')}")
    elif rc:
        gpu_fallbacks = rc.get("gpu_fallbacks") or rc.get("ordered")
        if gpu_fallbacks and isinstance(gpu_fallbacks, list):
            sky_conf.setdefault("resources", {})["ordered"] = gpu_fallbacks
            sky_conf["resources"].pop("any_of", None)

    tmp = f"/tmp/sky_tabular_serve_{serve_run_id}.yaml"
    with open(tmp, "w") as fh:
        _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

    task = sky.Task.from_yaml(tmp)
    task_envs = {
        "SERVE_RUN_ID":        serve_run_id,
        "REGISTRY_MODEL_NAME": registry_model,
        "MODEL_ALIAS":         alias,
        "SERVE_PORT":          str(serve_port),
    }
    if mlflow_tracking_uri:
        task_envs["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        print("[serve] MLFLOW_TRACKING_URI override is set for remote replicas")
    else:
        print(
            "[serve][warn] MLFLOW_TRACKING_URI override is empty; "
            "replicas will use params/config value."
        )
    task_envs.update(_aws_runtime_envs())
    task.update_envs(task_envs)

    service_name = _k8s_name(serve_run_id, "tabserv", max_len=32)
    print(f"Launching sky serve service: {service_name}")
    sky.stream_and_get(sky.serve.up(task, service_name=service_name))
    print(f"Tabular serve service launched: {service_name}")
    _xcom_push({"service_name": service_name, "serve_port": serve_port})


def wait_tabular_serve():
    """Poll sky.serve.status() until the service is READY and return the endpoint URL."""
    import sky

    info         = json.loads(_env("SERVICE_INFO_JSON") or "{}")
    service_name = info.get("service_name", "")
    timeout      = int(_env("TABULAR_SERVE_HEALTH_TIMEOUT_SECONDS", "600"))
    interval     = 20

    if not service_name:
        raise RuntimeError("SERVICE_INFO_JSON must contain 'service_name'")

    print(f"Polling sky serve status for service '{service_name}' (timeout={timeout}s)")
    live_streamer = _SkyServeLiveStreamer(
        service_name=service_name,
        max_replicas=_SERVE_LOG_MAX_REPLICAS,
        include_controller=_SERVE_STREAM_CONTROLLER,
        include_load_balancer=_SERVE_STREAM_LOAD_BALANCER,
    )
    live_streamer.start_base_streams()
    live_streamer.refresh_replica_streams()
    print(
        "[serve-live] streaming replica logs only "
        f"(follow_only=True, replicas={_SERVE_LOG_MAX_REPLICAS}, "
        f"controller={_SERVE_STREAM_CONTROLLER}, "
        f"load_balancer={_SERVE_STREAM_LOAD_BALANCER})"
    )

    elapsed = 0
    try:
        while elapsed < timeout:
            try:
                statuses = sky.get(sky.serve.status(service_names=[service_name]))
                if statuses:
                    svc = statuses[0]
                    if isinstance(svc, dict):
                        endpoint  = svc.get("endpoint_url") or svc.get("endpoint", "")
                        status_val = str(svc.get("status", "")).upper()
                    else:
                        endpoint  = getattr(svc, "endpoint_url", None) or getattr(svc, "endpoint", "")
                        status_val = str(getattr(svc, "status", "")).upper()
                    if "READY" in status_val and endpoint:
                        print(f"Tabular serve healthy after {elapsed}s: {endpoint}")
                        _xcom_push(endpoint)
                        return
                    if "FAILED" in status_val:
                        raise RuntimeError(f"Service '{service_name}' entered FAILED state.")
                    print(f"[{elapsed}s] Status={status_val}, endpoint={endpoint or '—'}")
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"[{elapsed}s] status check error: {exc}")

            live_streamer.refresh_replica_streams()

            time.sleep(interval)
            elapsed += interval

        raise RuntimeError(
            f"Tabular serve service '{service_name}' not READY after {timeout}s."
        )
    finally:
        live_streamer.stop()


def register_endpoint():
    import boto3
    from datetime import datetime

    endpoint_url  = _env("ENDPOINT_URL")
    serve_run_id  = _env("SERVE_RUN_ID")
    model_id      = _env("HF_MODEL_ID")
    # SERVICE_NAME (sky.serve pattern) with fallback to CLUSTER_NAME (sky.launch pattern)
    service_name  = _env("SERVICE_NAME", "") or _env("CLUSTER_NAME", "")
    orchestration = _env("ORCHESTRATION", "vllm_single_node")
    bucket        = _env("S3_BUCKET", "k8s-mlops-platform-bucket")

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
    if service_name:
        payload["service_name"] = service_name
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode(),
        ContentType="application/json",
    )
    print(f"Endpoint registered: {endpoint_url} → s3://{bucket}/{key}")


def update_tabular_serve():
    """Rolling-update an existing SkyPilot sky serve service to pick up a new champion model.

    Calls sky.serve.update() with the same YAML used at launch.  New replicas
    restart and app.py loads mlflow.pyfunc via MODEL_ALIAS (default "champion"),
    picking up whatever version is currently aliased @champion in MLflow.
    Zero-downtime — SkyServe keeps old replicas serving until new ones are READY.
    """
    import sky
    import yaml as _yaml

    service_name   = _env("SERVICE_NAME")
    registry_model = _env("REGISTRY_MODEL_NAME")
    alias          = _env("MODEL_ALIAS", "champion")
    serve_port     = int(_env("SERVE_PORT", "8000"))
    num_nodes      = int(_env("NUM_NODES", "1"))
    rc             = _rc()

    if not service_name:
        raise RuntimeError("SERVICE_NAME env var is required for update-tabular-serve")

    kind = "tabular_serve_multi" if num_nodes > 1 else "tabular_serve"
    yaml_path = _yaml_path(kind, rc)
    print(f"Updating sky serve service '{service_name}' using YAML: {yaml_path}  (num_nodes={num_nodes})")

    _apply_serve_controller_config()

    with open(yaml_path) as fh:
        sky_conf = _yaml.safe_load(fh)

    service_cfg = sky_conf.setdefault("service", {})
    service_cfg["ports"] = int(serve_port)

    if num_nodes > 1:
        sky_conf["num_nodes"] = num_nodes

    if rc:
        gpu_fallbacks = rc.get("gpu_fallbacks") or rc.get("ordered")
        if gpu_fallbacks and isinstance(gpu_fallbacks, list):
            sky_conf.setdefault("resources", {})["ordered"] = gpu_fallbacks
            sky_conf["resources"].pop("any_of", None)

    tmp = f"/tmp/sky_tabular_update_{service_name}.yaml"
    with open(tmp, "w") as fh:
        _yaml.dump(sky_conf, fh, default_flow_style=False, allow_unicode=True)

    task = sky.Task.from_yaml(tmp)
    task.update_envs({
        "REGISTRY_MODEL_NAME": registry_model,
        "MODEL_ALIAS":         alias,
        "SERVE_PORT":          str(serve_port),
    })

    print(f"Triggering rolling update for service '{service_name}' → models:/{registry_model}@{alias}")
    sky.stream_and_get(sky.serve.update(task, service_name=service_name))
    print(f"Rolling update complete: '{service_name}'")
    _xcom_push({"service_name": service_name, "status": "updated"})


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
    "launch-ray-vllm":       launch_ray_vllm,
    "wait-ray-vllm":         wait_ray_vllm,
    # ── Tabular sky serve (sky.serve.up — managed replicas) ───────────────────
    "launch-tabular-serve":  launch_tabular_serve,
    "wait-tabular-serve":    wait_tabular_serve,
    "update-tabular-serve":  update_tabular_serve,
    "register-endpoint": register_endpoint,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        cmds = " | ".join(_COMMANDS)
        print(f"Usage: sky_runner.py [{cmds}]", file=sys.stderr)
        sys.exit(1)
    _COMMANDS[sys.argv[1]]()
