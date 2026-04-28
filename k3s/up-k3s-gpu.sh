#!/usr/bin/env bash
set -euo pipefail

# Bring up local k3s with Docker runtime and NVIDIA GPU support.
# Designed so you can run k3s/deploy.sh afterwards using k3s kubeconfig.
#
# Usage:
#   sudo bash k3s/up-k3s-gpu.sh
#
# Optional env vars:
#   INSTALL_K3S_VERSION=v1.34.1+k3s1
#   DEVICE_PLUGIN_VERSION=v0.17.1
#   K3S_HTTPS_PORT=16443
#   WAIT_TIMEOUT=240s
#   FORCE_REINSTALL_K3S=0
#   RUN_GPU_SMOKE=1
#   TEST_POD_IMAGE=takenking9879/ray-train-ssh:2.53.0

INSTALL_K3S_VERSION="${INSTALL_K3S_VERSION:-v1.34.1+k3s1}"
DEVICE_PLUGIN_VERSION="${DEVICE_PLUGIN_VERSION:-v0.17.1}"
DEVICE_PLUGIN_MANIFEST="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${DEVICE_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml"
K3S_HTTPS_PORT="${K3S_HTTPS_PORT:-16443}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-240s}"
FORCE_REINSTALL_K3S="${FORCE_REINSTALL_K3S:-0}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
TEST_POD_IMAGE="${TEST_POD_IMAGE:-takenking9879/ray-train-ssh:2.53.0}"
TEST_POD_NAME="${TEST_POD_NAME:-nvidia-smi-k3s-up}"
K3S_KUBECONFIG="/etc/rancher/k3s/k3s.yaml"
DESKTOP_DOCKER_HOST="unix:///mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock"

log() {
  echo "[up-k3s-gpu] $*"
}

run() {
  log "RUN: $*"
  bash -lc "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

for cmd in bash curl awk sed grep ss systemctl docker; do
  need_cmd "${cmd}"
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

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

  log "Detected Docker Desktop daemon. Installing/enabling local Docker engine in WSL."
  run "apt-get update"
  run "apt-get install -y docker.io"
  run "systemctl enable --now docker"
  if grep -q '/Docker/host' /proc/mounts; then
    run "umount /Docker/host || true"
  fi

  os_name="$(docker_os)"
  if [[ "${os_name}" == *"Docker Desktop"* ]]; then
    echo "Failed to switch away from Docker Desktop daemon." >&2
    exit 1
  fi
  log "Using local Docker engine: ${os_name}"
}

ensure_test_image_present() {
  if docker image inspect "${TEST_POD_IMAGE}" >/dev/null 2>&1; then
    return
  fi

  log "Image ${TEST_POD_IMAGE} not found locally. Trying import from Docker Desktop daemon."
  if DOCKER_HOST="${DESKTOP_DOCKER_HOST}" docker image inspect "${TEST_POD_IMAGE}" >/dev/null 2>&1; then
    run "DOCKER_HOST='${DESKTOP_DOCKER_HOST}' docker save '${TEST_POD_IMAGE}' | docker load"
    return
  fi

  log "Image not found in Desktop daemon. Pulling from registry."
  run "docker pull '${TEST_POD_IMAGE}'"
}

wait_for_gpu_allocatable() {
  local wait_seconds start_ts now_ts alloc_line
  wait_seconds="${WAIT_TIMEOUT%s}"
  if [[ -z "${wait_seconds}" || "${wait_seconds}" == "${WAIT_TIMEOUT}" ]]; then
    wait_seconds=240
  fi
  start_ts="$(date +%s)"

  while true; do
    alloc_line="$(kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" alloc="}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}')"
    log "${alloc_line}"
    if grep -Eq 'alloc=[1-9][0-9]*' <<<"${alloc_line}"; then
      return 0
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= wait_seconds )); then
      return 1
    fi
    sleep 3
  done
}

log "Step 0: ensure local Docker engine"
ensure_local_docker_engine

log "Step 1: ensure NVIDIA Docker runtime"
run "command -v nvidia-ctk >/dev/null 2>&1 || (apt-get update && apt-get install -y nvidia-container-toolkit)"
run "nvidia-ctk runtime configure --runtime=docker --set-as-default"
run "systemctl restart docker"
run "docker info | sed -n '/Runtimes:/,/Default Runtime:/p'"

log "Step 2: ensure test image exists (for optional smoke test)"
ensure_test_image_present

if [[ "${FORCE_REINSTALL_K3S}" == "1" ]] && command -v k3s-uninstall.sh >/dev/null 2>&1; then
  log "FORCE_REINSTALL_K3S=1 -> uninstalling current k3s"
  run "k3s-uninstall.sh || true"
fi

if ! command -v k3s >/dev/null 2>&1; then
  if ss -lnt "( sport = :${K3S_HTTPS_PORT} )" | grep -q ":${K3S_HTTPS_PORT}"; then
    echo "Port ${K3S_HTTPS_PORT} is already in use. Set K3S_HTTPS_PORT to a free port." >&2
    exit 1
  fi
  log "Step 3: installing k3s --docker"
  run "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='${INSTALL_K3S_VERSION}' INSTALL_K3S_EXEC='server --docker --disable traefik --write-kubeconfig-mode 644 --https-listen-port ${K3S_HTTPS_PORT}' sh -"
else
  log "Step 3: k3s already installed, reusing installation"
fi

log "Step 4: start k3s and wait for node Ready"
run "systemctl enable --now k3s"
run "systemctl is-active k3s"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=condition=Ready node --all --timeout='${WAIT_TIMEOUT}'"

NODE_NAME="$(kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "${NODE_NAME}" ]]; then
  echo "k3s node not found." >&2
  exit 1
fi
log "Node: ${NODE_NAME}"

log "Step 5: deploy NVIDIA device plugin"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '${DEVICE_PLUGIN_MANIFEST}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout='${WAIT_TIMEOUT}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system wait --for=condition=Ready pod -l name=nvidia-device-plugin-ds --timeout='${WAIT_TIMEOUT}' || true"

log "Step 6: wait for nvidia.com/gpu allocatable"
if ! wait_for_gpu_allocatable; then
  log "GPU resource did not appear in time. Diagnostics:"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' describe node '${NODE_NAME}' | sed -n '1,260p'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system get pods -o wide | grep -Ei 'nvidia|device-plugin|kube' || true"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system logs daemonset/nvidia-device-plugin-daemonset --all-containers=true --tail=300 || true"
  exit 2
fi

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  log "Step 7: GPU smoke pod (nvidia-smi)"
  cat >/tmp/${TEST_POD_NAME}.yaml <<YAML
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
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '/tmp/${TEST_POD_NAME}.yaml'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=jsonpath='{.status.phase}'=Succeeded pod/${TEST_POD_NAME} --timeout='${WAIT_TIMEOUT}' || true"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' logs '${TEST_POD_NAME}' --tail=120"
fi

log "k3s + GPU ready."
log "Next:"
log "  KUBECONFIG=${K3S_KUBECONFIG} ./k3s/deploy.sh"
log "To stop cluster:"
log "  sudo bash k3s/down-k3s-gpu.sh"
