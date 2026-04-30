#!/usr/bin/env bash
set -euo pipefail

# Start local k3s cluster (and Docker). If k3s is missing, install it first.
#
# Usage:
#   sudo bash k3s/up-k3s-gpu.sh
#
# Optional env vars:
#   START_LOCAL_DOCKER=1
#   WAIT_TIMEOUT=180s
#   INSTALL_K3S_VERSION=v1.34.1+k3s1
#   K3S_HTTPS_PORT=16443

START_LOCAL_DOCKER="${START_LOCAL_DOCKER:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180s}"
INSTALL_K3S_VERSION="${INSTALL_K3S_VERSION:-v1.34.1+k3s1}"
K3S_HTTPS_PORT="${K3S_HTTPS_PORT:-16443}"
K3S_KUBECONFIG="${K3S_KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

log() {
  echo "[up-k3s-gpu] $*"
}

run() {
  log "RUN: $*"
  bash -lc "$*"
}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

if [[ "${START_LOCAL_DOCKER}" == "1" ]]; then
  if systemctl list-unit-files | grep -q '^docker\.service'; then
    log "Starting Docker service"
    run "systemctl start docker"
    run "systemctl start docker.socket || true"
  else
    log "Docker service not found. Continuing (k3s might be configured with another runtime)."
  fi
fi

if ! systemctl list-unit-files | grep -q '^k3s\.service'; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install k3s." >&2
    exit 1
  fi

  if command -v ss >/dev/null 2>&1 && ss -lnt "( sport = :${K3S_HTTPS_PORT} )" | grep -q ":${K3S_HTTPS_PORT}"; then
    echo "Port ${K3S_HTTPS_PORT} is already in use. Set K3S_HTTPS_PORT to a free port." >&2
    exit 1
  fi

  log "k3s not found. Installing k3s ${INSTALL_K3S_VERSION}"
  run "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='${INSTALL_K3S_VERSION}' INSTALL_K3S_EXEC='server --docker --disable traefik --write-kubeconfig-mode 644 --https-listen-port ${K3S_HTTPS_PORT}' sh -"
fi

log "Starting k3s service"
run "systemctl start k3s"
run "systemctl is-active k3s"

if command -v kubectl >/dev/null 2>&1 && [[ -f "${K3S_KUBECONFIG}" ]]; then
  log "Waiting for node Ready"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=condition=Ready node --all --timeout='${WAIT_TIMEOUT}'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get nodes -o wide"
else
  log "Skipping kubectl readiness check (kubectl or kubeconfig missing)."
fi

log "Done. k3s is up."
