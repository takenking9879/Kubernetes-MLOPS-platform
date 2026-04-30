#!/usr/bin/env bash
set -euo pipefail

# Stop local k3s cluster and (optionally) Docker without uninstalling.
#
# Usage:
#   sudo bash k3s/down-k3s-gpu.sh
#
# Optional env vars:
#   STOP_LOCAL_DOCKER=1   # set to 0 if you want to keep Docker running

STOP_LOCAL_DOCKER="${STOP_LOCAL_DOCKER:-1}"

log() {
  echo "[down-k3s-gpu] $*"
}

run() {
  log "RUN: $*"
  bash -lc "$*"
}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

log "Stopping k3s service (without uninstall)"
run "systemctl stop k3s || true"

if [[ "${STOP_LOCAL_DOCKER}" == "1" ]]; then
  log "Stopping local Docker daemon"
  run "systemctl stop docker.socket || true"
  run "systemctl stop docker || true"
fi

log "Done. k3s is down (installation preserved)."
