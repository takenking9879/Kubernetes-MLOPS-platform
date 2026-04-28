#!/usr/bin/env bash
set -euo pipefail

# Reproducible GPU enablement for k3s using Docker runtime (cri-dockerd).
# Target: run a Kubernetes pod with nvidia-smi using k3s --docker.
#
# Usage:
#   bash k3s/sky/k3s-docker-gpu-pod-repro.sh
#
# Optional env vars:
#   INSTALL_K3S_VERSION=v1.34.1+k3s1
#   DEVICE_PLUGIN_VERSION=v0.17.1
#   TEST_POD_IMAGE=takenking9879/ray-train-ssh:2.53.0
#   TEST_POD_NAME=nvidia-smi-k3s
#   WAIT_TIMEOUT=240s
#   SKIP_K3S_INSTALL=0
#   FORCE_REINSTALL_K3S=0
#   K3S_HTTPS_PORT=16443

INSTALL_K3S_VERSION="${INSTALL_K3S_VERSION:-v1.34.1+k3s1}"
DEVICE_PLUGIN_VERSION="${DEVICE_PLUGIN_VERSION:-v0.17.1}"
DEVICE_PLUGIN_MANIFEST="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${DEVICE_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml"
TEST_POD_IMAGE="${TEST_POD_IMAGE:-takenking9879/ray-train-ssh:2.53.0}"
TEST_POD_NAME="${TEST_POD_NAME:-nvidia-smi-k3s}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-240s}"
SKIP_K3S_INSTALL="${SKIP_K3S_INSTALL:-0}"
FORCE_REINSTALL_K3S="${FORCE_REINSTALL_K3S:-0}"
K3S_HTTPS_PORT="${K3S_HTTPS_PORT:-16443}"
REPORT="/tmp/k3s-docker-gpu-repro-$(date +%Y%m%d-%H%M%S).log"
TEST_POD_FILE="/tmp/k3s-gpu-test-pod.yaml"

K3S_KUBECONFIG="/etc/rancher/k3s/k3s.yaml"
DESKTOP_DOCKER_HOST="unix:///mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock"

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "${REPORT}"
}

run() {
  local cmd="$1"
  log "HOST $ ${cmd}"
  bash -lc "${cmd}" 2>&1 | tee -a "${REPORT}"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

docker_os() {
  docker info --format '{{.OperatingSystem}}' 2>/dev/null || true
}

ensure_local_docker_engine() {
  local os_name
  os_name="$(docker_os)"
  if [[ "${os_name}" != *"Docker Desktop"* ]]; then
    log "Docker engine is local: ${os_name}"
    return
  fi

  log "Detected Docker Desktop daemon. Switching to local WSL Docker engine."
  run "${SUDO} apt-get update"
  run "${SUDO} apt-get install -y docker.io"
  run "${SUDO} systemctl enable --now docker"
  if grep -q '/Docker/host' /proc/mounts; then
    run "${SUDO} umount /Docker/host || true"
  fi

  os_name="$(docker_os)"
  if [[ "${os_name}" == *"Docker Desktop"* ]]; then
    log "Failed to switch away from Docker Desktop daemon."
    exit 5
  fi
  log "Using local Docker engine: ${os_name}"
}

ensure_test_image_present() {
  if docker image inspect "${TEST_POD_IMAGE}" >/dev/null 2>&1; then
    return
  fi

  log "Image ${TEST_POD_IMAGE} not found locally. Trying import from Docker Desktop daemon."
  if ${SUDO} DOCKER_HOST="${DESKTOP_DOCKER_HOST}" docker image inspect "${TEST_POD_IMAGE}" >/dev/null 2>&1; then
    run "${SUDO} DOCKER_HOST='${DESKTOP_DOCKER_HOST}' docker save '${TEST_POD_IMAGE}' | docker load"
    return
  fi

  log "Image not present in Docker Desktop daemon either. Pulling from registry."
  run "docker pull '${TEST_POD_IMAGE}'"
}

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

cleanup() {
  if command -v kubectl >/dev/null 2>&1; then
    kubectl --kubeconfig "${K3S_KUBECONFIG}" delete pod "${TEST_POD_NAME}" --ignore-not-found=true >/dev/null 2>&1 || true
  fi
  rm -f "${TEST_POD_FILE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for cmd in bash curl kubectl docker awk sed grep; do
  require_cmd "${cmd}"
done

log "Report file: ${REPORT}"

if [[ "${EUID}" -ne 0 ]]; then
  if ! sudo -n true >/dev/null 2>&1; then
    cat <<MSG | tee -a "${REPORT}"
[precheck] This script needs root privileges (k3s install, docker runtime config).
Run it with:
  sudo TEST_POD_IMAGE=${TEST_POD_IMAGE} bash k3s/sky/k3s-docker-gpu-pod-repro.sh
or configure passwordless sudo for this session.
MSG
    exit 3
  fi
fi

log "Step 0/8: ensure local Docker engine + image availability"
ensure_local_docker_engine
ensure_test_image_present

log "Step 1/8: verify Docker GPU access on host"
run "docker run --rm --gpus all --entrypoint bash '${TEST_POD_IMAGE}' -lc 'nvidia-smi'"

log "Step 2/8: configure NVIDIA runtime as Docker default"
run "command -v nvidia-ctk >/dev/null 2>&1 || (${SUDO} apt-get update && ${SUDO} apt-get install -y nvidia-container-toolkit)"
run "${SUDO} nvidia-ctk runtime configure --runtime=docker --set-as-default"
if systemctl list-unit-files docker.service >/dev/null 2>&1; then
  run "${SUDO} systemctl restart docker"
else
  log "docker.service not present (Docker Desktop context). Skipping daemon restart."
fi
run "docker info | sed -n '/Runtimes:/,/Default Runtime:/p'"

log "Step 3/8: install or reinstall k3s with --docker"
NEED_INSTALL=0
if ! command -v k3s >/dev/null 2>&1 || [[ "${FORCE_REINSTALL_K3S}" == "1" ]]; then
  NEED_INSTALL=1
fi

if [[ "${FORCE_REINSTALL_K3S}" == "1" ]] && command -v k3s-uninstall.sh >/dev/null 2>&1; then
  run "${SUDO} k3s-uninstall.sh || true"
fi

if [[ "${NEED_INSTALL}" == "1" ]] && ss -lnt "( sport = :${K3S_HTTPS_PORT} )" | grep -q ":${K3S_HTTPS_PORT}"; then
  log "Port ${K3S_HTTPS_PORT} is already in use. Set K3S_HTTPS_PORT to a free port and rerun."
  exit 4
fi

if command -v k3s >/dev/null 2>&1 && [[ "${FORCE_REINSTALL_K3S}" != "1" ]]; then
  if [[ "${SKIP_K3S_INSTALL}" == "1" ]]; then
    log "k3s already present and SKIP_K3S_INSTALL=1, skipping installation."
  else
    log "k3s already present. Reusing current installation (set FORCE_REINSTALL_K3S=1 to reinstall)."
  fi
else
  run "curl -sfL https://get.k3s.io | ${SUDO} INSTALL_K3S_VERSION='${INSTALL_K3S_VERSION}' INSTALL_K3S_EXEC='server --docker --disable traefik --write-kubeconfig-mode 644 --https-listen-port ${K3S_HTTPS_PORT}' sh -"
fi

log "Step 4/8: wait for k3s service and node readiness"
run "${SUDO} systemctl enable --now k3s"
run "${SUDO} systemctl is-active k3s"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get nodes -o wide"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=condition=Ready node --all --timeout='${WAIT_TIMEOUT}'"

NODE_NAME="$(kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "${NODE_NAME}" ]]; then
  echo "No k3s node detected." >&2
  exit 1
fi
log "Detected k3s node: ${NODE_NAME}"

log "Step 5/8: deploy NVIDIA device plugin"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '${DEVICE_PLUGIN_MANIFEST}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout='${WAIT_TIMEOUT}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system wait --for=condition=Ready pod -l name=nvidia-device-plugin-ds --timeout='${WAIT_TIMEOUT}' || true"

log "Step 6/8: verify allocatable nvidia.com/gpu"
WAIT_SECONDS="${WAIT_TIMEOUT%s}"
if [[ -z "${WAIT_SECONDS}" || "${WAIT_SECONDS}" == "${WAIT_TIMEOUT}" ]]; then
  WAIT_SECONDS=240
fi
START_TS="$(date +%s)"
ALLOC_LINE=""
while true; do
  ALLOC_LINE="$(kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" allocatable="}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}')"
  log "${ALLOC_LINE}"
  if grep -Eq 'allocatable=[1-9][0-9]*' <<<"${ALLOC_LINE}"; then
    break
  fi
  NOW_TS="$(date +%s)"
  if (( NOW_TS - START_TS >= WAIT_SECONDS )); then
    break
  fi
  sleep 3
done

if ! grep -Eq 'allocatable=[1-9][0-9]*' <<<"${ALLOC_LINE}"; then
  log "GPU not allocatable yet. Collecting diagnostics."
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' describe node '${NODE_NAME}' | sed -n '1,260p'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system get pods -o wide | grep -Ei 'nvidia|device-plugin|kube' || true"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system logs daemonset/nvidia-device-plugin-daemonset --all-containers=true --tail=300 || true"
  run "${SUDO} journalctl -u k3s --no-pager -n 300 || true"
  log "Final status: FAILED (k3s node has no allocatable nvidia.com/gpu)."
  log "See report: ${REPORT}"
  exit 2
fi

log "Step 7/8: run GPU pod smoke test"
cat >"${TEST_POD_FILE}" <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${TEST_POD_NAME}
spec:
  restartPolicy: Never
  containers:
  - name: gpu
    image: ${TEST_POD_IMAGE}
    command: ["bash", "-lc", "nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
YAML

run "kubectl --kubeconfig '${K3S_KUBECONFIG}' delete pod '${TEST_POD_NAME}' --ignore-not-found=true"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '${TEST_POD_FILE}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=jsonpath='{.status.phase}'=Succeeded pod/${TEST_POD_NAME} --timeout='${WAIT_TIMEOUT}' || true"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get pod '${TEST_POD_NAME}' -o wide"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' logs '${TEST_POD_NAME}' --tail=200"

log "Step 8/8: final verification"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get node '${NODE_NAME}' -o jsonpath='{.metadata.name}{\" alloc=\"}{.status.allocatable.nvidia\\.com/gpu}{\" cap=\"}{.status.capacity.nvidia\\.com/gpu}{\"\\n\"}'"

log "Final status: SUCCESS (k3s GPU pod executed nvidia-smi)."
log "See report: ${REPORT}"
