#!/usr/bin/env bash
set -euo pipefail

# Tear down local k3s cluster created for GPU runtime tests.
#
# Usage:
#   sudo bash k3s/down-k3s-gpu.sh
#
# Optional env vars:
#   STOP_LOCAL_DOCKER=0   # set to 1 to stop docker.service after removing k3s

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

log "Stopping and uninstalling k3s (if present)"
if command -v k3s-uninstall.sh >/dev/null 2>&1; then
  run "k3s-uninstall.sh || true"
else
  run "systemctl stop k3s || true"
  run "systemctl disable k3s || true"
  run "rm -f /etc/systemd/system/k3s.service /etc/systemd/system/k3s.service.env || true"
  run "systemctl daemon-reload || true"
fi

log "Cleaning leftover kube config links (if any)"
run "rm -f /usr/local/bin/kubectl /usr/local/bin/crictl /usr/local/bin/ctr || true"

if [[ "${STOP_LOCAL_DOCKER}" == "1" ]]; then
  log "Stopping local Docker daemon"
  run "systemctl stop docker.socket || true"
  run "systemctl stop docker || true"
  run "systemctl disable docker.socket || true"
  run "systemctl disable docker || true"
fi

log "Done. k3s is down."
