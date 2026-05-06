#!/bin/bash
# SkyPilot runner entrypoint — writes cloud credential files from env vars.
#
# All API keys come from env-secret (--from-env-file=.env) injected via envFrom.
# This avoids mounting individual secrets per provider.
#
# Required env vars (add to .env):
#   AWS_ACCESS_KEY_ID        — boto3 reads natively, no file needed
#   AWS_SECRET_ACCESS_KEY    — boto3 reads natively, no file needed
#   AWS_DEFAULT_REGION       — optional, defaults to us-east-1
#   RUNPOD_API_KEY           — written to ~/.runpod/config.toml
#   VAST_API_KEY / VASTAI_API_KEY — written to ~/.config/vastai/vast_api_key
#   SKY_JOBS_CONTROLLER_DISK_SIZE — optional (default: 30); must be <= 40 on RunPod
set -e

# ── SkyPilot config (managed jobs controller sizing) ─────────────────────────
# RunPod rejects controller pods with disk_size > 40 GB.
# Merge a safe default into ~/.sky/config.yaml while preserving existing fields.
SKY_JOBS_CONTROLLER_DISK_SIZE="${SKY_JOBS_CONTROLLER_DISK_SIZE:-30}"
mkdir -p ~/.sky
python3 - <<'PY'
import os
from pathlib import Path

import yaml

config_path = Path.home() / '.sky' / 'config.yaml'
disk_size = int(os.getenv('SKY_JOBS_CONTROLLER_DISK_SIZE', '30'))

if disk_size > 40:
    print(
        f"[entrypoint] WARNING: SKY_JOBS_CONTROLLER_DISK_SIZE={disk_size} exceeds RunPod cap 40; forcing 40"
    )
    disk_size = 40

config = {}
if config_path.exists():
    try:
        loaded = yaml.safe_load(config_path.read_text())
        if isinstance(loaded, dict):
            config = loaded
    except Exception as exc:
        print(f"[entrypoint] WARNING: could not parse ~/.sky/config.yaml ({exc}); rewriting minimal config")

jobs = config.setdefault('jobs', {})
controller = jobs.setdefault('controller', {})
resources = controller.setdefault('resources', {})
resources['disk_size'] = disk_size

kubernetes = config.setdefault('kubernetes', {})
pod_config = kubernetes.setdefault('pod_config', {})
spec = pod_config.setdefault('spec', {})
containers = spec.setdefault('containers', [])

ray_container = next(
    (container for container in containers if container.get('name') == 'ray-node'),
    None,
)
if ray_container is None:
    containers.append({
        'name': 'ray-node',
        'imagePullPolicy': 'IfNotPresent',
    })
else:
    ray_container['imagePullPolicy'] = 'IfNotPresent'

config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
print(f"[entrypoint] SkyPilot jobs controller disk_size set to {disk_size} GB")
PY

# ── RunPod ─────────────────────────────────────────────────────────────────────
# SkyPilot reads ~/.runpod/config.toml via Python tomllib.
# Format: [default] profile with api_key.
if [ -n "${RUNPOD_API_KEY}" ]; then
    mkdir -p ~/.runpod
    cat > ~/.runpod/config.toml << EOF
[default]
api_key = "${RUNPOD_API_KEY}"
EOF
    echo "[entrypoint] RunPod credentials written to ~/.runpod/config.toml"
else
    echo "[entrypoint] WARNING: RUNPOD_API_KEY not set — RunPod provider unavailable"
fi

# ── VastAI ─────────────────────────────────────────────────────────────────────
# SkyPilot reads ~/.config/vastai/vast_api_key (plain text, single line).
VAST_KEY="${VAST_API_KEY:-${VASTAI_API_KEY:-}}"
if [ -n "${VAST_KEY}" ]; then
    mkdir -p ~/.config/vastai
    printf '%s' "${VAST_KEY}" > ~/.config/vastai/vast_api_key
    echo "[entrypoint] VastAI credentials written to ~/.config/vastai/vast_api_key"
else
    echo "[entrypoint] WARNING: VAST_API_KEY/VASTAI_API_KEY not set — VastAI provider unavailable"
fi

# ── AWS ────────────────────────────────────────────────────────────────────────
# boto3 reads AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env natively.
# No file setup needed. SkyPilot's AWS backend uses boto3.
if [ -z "${AWS_ACCESS_KEY_ID}" ]; then
    echo "[entrypoint] WARNING: AWS_ACCESS_KEY_ID not set — AWS provider unavailable"
fi

exec "$@"
