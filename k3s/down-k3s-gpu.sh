#!/usr/bin/env bash
set -euo pipefail

# Stop local k3s and optional runtime services, preserving installation and images.
#
# Usage:
#   sudo bash k3s/down-k3s-gpu.sh
#
# Optional env vars:
#   STOP_K3S=1
#   CLEAN_K3S_CONTAINERS=1
#   STOP_K3S_EMBEDDED_CONTAINERD=1
#   STOP_HOST_CONTAINERD=1
#   STOP_LOCAL_DOCKER=1
#   STOP_DOCKER_DESKTOP=0
#   UNMOUNT_DOCKER_HOST=1
#   STOP_WAIT_TIMEOUT=45
#   CLEAN_PLATFORM_STATE=1
#   PLATFORM_NAMESPACES="kafka ray spark monitoring apps airflow skypilot"
#   NAMESPACE_DELETE_TIMEOUT=180s

STOP_K3S="${STOP_K3S:-1}"
CLEAN_K3S_CONTAINERS="${CLEAN_K3S_CONTAINERS:-1}"
STOP_K3S_EMBEDDED_CONTAINERD="${STOP_K3S_EMBEDDED_CONTAINERD:-1}"
STOP_HOST_CONTAINERD="${STOP_HOST_CONTAINERD:-1}"
STOP_LOCAL_DOCKER="${STOP_LOCAL_DOCKER:-1}"
STOP_DOCKER_DESKTOP="${STOP_DOCKER_DESKTOP:-0}"
UNMOUNT_DOCKER_HOST="${UNMOUNT_DOCKER_HOST:-1}"
STOP_WAIT_TIMEOUT="${STOP_WAIT_TIMEOUT:-45}"
CLEAN_PLATFORM_STATE="${CLEAN_PLATFORM_STATE:-1}"
PLATFORM_NAMESPACES="${PLATFORM_NAMESPACES:-kafka ray spark monitoring apps airflow skypilot}"
NAMESPACE_DELETE_TIMEOUT="${NAMESPACE_DELETE_TIMEOUT:-180s}"
K3S_KUBECONFIG="${K3S_KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

K3S_SERVICE_ENV_FILE="/etc/systemd/system/k3s.service.env"
K3S_SERVICE_FILE="/etc/systemd/system/k3s.service"

log() {
  echo "[down-k3s-gpu] $*"
}

run() {
  log "RUN: $*"
  bash -lc "$*"
}

unit_exists() {
  local unit="$1"
  systemctl list-unit-files "${unit}" --no-legend 2>/dev/null | awk '{print $1}' | grep -Fxq "${unit}"
}

is_wsl() {
  [[ -n "${WSL_INTEROP:-}" ]] || grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null
}

wait_for_unit_inactive() {
  local unit="$1"
  local elapsed=0
  while (( elapsed < STOP_WAIT_TIMEOUT )); do
    if ! systemctl is-active --quiet "${unit}"; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
}

stop_unit_if_present() {
  local unit="$1"
  if ! unit_exists "${unit}"; then
    log "Unit ${unit} not found. Skipping."
    return
  fi
  log "Stopping ${unit}"
  run "systemctl stop ${unit} || true"
  if ! wait_for_unit_inactive "${unit}"; then
    echo "Timed out waiting for ${unit} to stop." >&2
    exit 1
  fi
}

k3s_uses_docker_runtime() {
  if [[ -f "${K3S_SERVICE_ENV_FILE}" ]] && grep -q -- '--docker' "${K3S_SERVICE_ENV_FILE}"; then
    return 0
  fi
  if [[ -f "${K3S_SERVICE_FILE}" ]] && grep -q -- '--docker' "${K3S_SERVICE_FILE}"; then
    return 0
  fi
  return 1
}

kube_cmd() {
  if command -v k3s >/dev/null 2>&1; then
    k3s kubectl --kubeconfig "${K3S_KUBECONFIG}" "$@"
    return
  fi
  kubectl --kubeconfig "${K3S_KUBECONFIG}" "$@"
}

cluster_api_reachable() {
  kube_cmd get --raw=/readyz >/dev/null 2>&1
}

cleanup_platform_state() {
  if [[ "${CLEAN_PLATFORM_STATE}" != "1" ]]; then
    log "Skipping cluster state cleanup (CLEAN_PLATFORM_STATE=${CLEAN_PLATFORM_STATE})"
    return
  fi

  if ! cluster_api_reachable; then
    log "k3s API is not reachable. Skipping namespace cleanup."
    return
  fi

  log "Cleaning platform namespaces before shutdown: ${PLATFORM_NAMESPACES}"
  for ns in ${PLATFORM_NAMESPACES}; do
    if kube_cmd get namespace "${ns}" >/dev/null 2>&1; then
      log "Deleting namespace ${ns}"
      kube_cmd delete namespace "${ns}" --ignore-not-found --wait=false || true
    fi
  done

  for ns in ${PLATFORM_NAMESPACES}; do
    if ! kube_cmd get namespace "${ns}" >/dev/null 2>&1; then
      continue
    fi

    if ! kube_cmd wait --for=delete "namespace/${ns}" --timeout="${NAMESPACE_DELETE_TIMEOUT}" >/dev/null 2>&1; then
      log "Namespace ${ns} is stuck terminating. Removing finalizers."
      kube_cmd patch namespace "${ns}" --type merge -p '{"metadata":{"finalizers":[]}}' >/dev/null 2>&1 || true
      kube_cmd wait --for=delete "namespace/${ns}" --timeout=30s >/dev/null 2>&1 || true
    fi
  done

  log "Platform namespace cleanup completed."
}

try_stop_docker_desktop() {
  if [[ "${STOP_DOCKER_DESKTOP}" != "1" ]]; then
    return
  fi
  if ! is_wsl; then
    return
  fi
  if ! command -v powershell.exe >/dev/null 2>&1; then
    return
  fi

  log "Trying to stop Docker Desktop from WSL"
  run "powershell.exe -NoProfile -Command \"Stop-Process -Name 'Docker Desktop' -Force -ErrorAction SilentlyContinue; Stop-Service -Name 'com.docker.service' -Force -ErrorAction SilentlyContinue\" >/dev/null 2>&1 || true"
}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

runtime_mode="containerd"
if k3s_uses_docker_runtime; then
  runtime_mode="docker"
fi

log "Detected k3s runtime mode: ${runtime_mode}"

cleanup_platform_state

if [[ "${STOP_K3S}" == "1" ]]; then
  stop_unit_if_present "k3s.service"
fi

if [[ "${CLEAN_K3S_CONTAINERS}" == "1" ]] && [[ -x "/usr/local/bin/k3s-killall.sh" ]]; then
  log "Running k3s-killall.sh (does not uninstall and should preserve image metadata/data directories)"
  run "/usr/local/bin/k3s-killall.sh"
fi

if [[ "${STOP_K3S_EMBEDDED_CONTAINERD}" == "1" ]]; then
  # Embedded k3s containerd dies with k3s service; this is just an explicit check.
  if pgrep -f '/run/k3s/containerd/containerd.sock|k3s.*containerd' >/dev/null 2>&1; then
    log "k3s embedded containerd process still present after stop; waiting briefly"
    sleep 2
    pgrep -a -f '/run/k3s/containerd/containerd.sock|k3s.*containerd' || true
  else
    log "k3s embedded containerd is down"
  fi
fi

if [[ "${STOP_HOST_CONTAINERD}" == "1" ]]; then
  stop_unit_if_present "containerd.service"
fi

if [[ "${STOP_LOCAL_DOCKER}" == "1" ]]; then
  stop_unit_if_present "docker.socket"
  stop_unit_if_present "docker.service"
  try_stop_docker_desktop
fi

if [[ "${UNMOUNT_DOCKER_HOST}" == "1" ]] && grep -q ' /Docker/host ' /proc/mounts 2>/dev/null; then
  log "Unmounting /Docker/host to avoid kubelet mount parser issues on next k3s start"
  run "umount /Docker/host || true"
fi

log "Done. k3s and selected dependencies are down."
log "Images imported into k3s containerd remain persisted under /var/lib/rancher/k3s unless you uninstall/clean data."
