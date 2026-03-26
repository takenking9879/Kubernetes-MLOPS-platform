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
#   VASTAI_API_KEY           — written to ~/.config/vastai/vast_api_key
set -e

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
if [ -n "${VASTAI_API_KEY}" ]; then
    mkdir -p ~/.config/vastai
    printf '%s' "${VASTAI_API_KEY}" > ~/.config/vastai/vast_api_key
    echo "[entrypoint] VastAI credentials written to ~/.config/vastai/vast_api_key"
else
    echo "[entrypoint] WARNING: VASTAI_API_KEY not set — VastAI provider unavailable"
fi

# ── AWS ────────────────────────────────────────────────────────────────────────
# boto3 reads AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env natively.
# No file setup needed. SkyPilot's AWS backend uses boto3.
if [ -z "${AWS_ACCESS_KEY_ID}" ]; then
    echo "[entrypoint] WARNING: AWS_ACCESS_KEY_ID not set — AWS provider unavailable"
fi

exec "$@"
