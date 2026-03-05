#!/bin/bash
# ============================================================
# RunPod GPU Worker Startup Script
# ============================================================
#
# Required environment variables:
#   RAY_HEAD_ADDRESS      — Ray GCS address, e.g. "100.x.x.x:6379" (Tailscale IP)
#                           or "10.244.x.x:6379" (pod IP for local testing)
#   TAILSCALE_AUTHKEY     — Ephemeral Tailscale auth key (one-use, generated per pod)
#
# Optional environment variables:
#   NUM_CPUS              — CPU cores to advertise to Ray (default: 10)
#   OBJECT_STORE_MEMORY   — Object store size in bytes (default: 8 GB)
#   RUNPOD_POD_ID         — Set automatically by RunPod; used as Tailscale hostname
#
# ============================================================
set -euo pipefail

# ── 1. Start Tailscale daemon ─────────────────────────────────────────────────
echo "[worker] Starting tailscaled..."
sudo tailscaled \
    --state=/tmp/tailscale.state \
    --socket=/tmp/tailscaled.sock \
    --tun=userspace-networking \
    &
TAILSCALED_PID=$!

# Give the daemon a moment to open its socket
sleep 2

# ── 2. Join the tailnet ───────────────────────────────────────────────────────
HOSTNAME="runpod-gpu-${RUNPOD_POD_ID:-$(hostname)}"
echo "[worker] Joining tailnet as ${HOSTNAME}..."
tailscale \
    --socket=/tmp/tailscaled.sock \
    up \
    --authkey="${TAILSCALE_AUTHKEY}" \
    --hostname="${HOSTNAME}" \
    --accept-routes

# ── 3. Resolve own Tailscale IP (advertised to Ray head for callbacks) ────────
MY_TS_IP=$(tailscale --socket=/tmp/tailscaled.sock ip -4)
echo "[worker] Tailscale IP: ${MY_TS_IP}"

# ── 4. Wait for Ray head GCS to be reachable ─────────────────────────────────
RAY_HEAD_HOST="${RAY_HEAD_ADDRESS%%:*}"
RAY_HEAD_PORT="${RAY_HEAD_ADDRESS##*:}"

echo "[worker] Waiting for Ray head at ${RAY_HEAD_HOST}:${RAY_HEAD_PORT}..."
until nc -z "${RAY_HEAD_HOST}" "${RAY_HEAD_PORT}" 2>/dev/null; do
    sleep 3
done
echo "[worker] Ray head is reachable."

# ── 5. Join the Ray cluster ───────────────────────────────────────────────────
echo "[worker] Joining Ray cluster at ${RAY_HEAD_ADDRESS}..."
exec ray start \
    --address="${RAY_HEAD_ADDRESS}" \
    --num-gpus=1 \
    --num-cpus="${NUM_CPUS:-10}" \
    --node-ip-address="${MY_TS_IP}" \
    --object-store-memory="${OBJECT_STORE_MEMORY:-8000000000}" \
    --min-worker-port=20000 \
    --max-worker-port=20020 \
    --block
