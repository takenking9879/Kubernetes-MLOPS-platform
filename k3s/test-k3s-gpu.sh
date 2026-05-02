#!/usr/bin/env bash
set -euo pipefail

# GPU smoke test for k3s using a local image already imported in k3s containerd.
#
# Usage:
#   sudo bash k3s/test-k3s-gpu.sh
#
# Optional env vars:
#   TEST_POD_IMAGE=takenking9879/ray-train:2.53.0
#   TEST_POD_NAME=nvidia-smi-k3s-local
#   WAIT_TIMEOUT=240s
#   K3S_KUBECONFIG=/etc/rancher/k3s/k3s.yaml
#   RUNTIME_CLASS_NAME=nvidia

TEST_POD_IMAGE="${TEST_POD_IMAGE:-takenking9879/ray-train:2.53.0}"
TEST_POD_NAME="${TEST_POD_NAME:-nvidia-smi-k3s-local}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-240s}"
K3S_KUBECONFIG="${K3S_KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
RUNTIME_CLASS_NAME="${RUNTIME_CLASS_NAME:-nvidia}"
TEST_POD_FILE="/tmp/${TEST_POD_NAME}.yaml"

log() {
  echo "[test-k3s-gpu] $*"
}

run() {
  log "RUN: $*"
  bash -lc "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

cleanup() {
  kubectl --kubeconfig "${K3S_KUBECONFIG}" delete pod "${TEST_POD_NAME}" --ignore-not-found=true >/dev/null 2>&1 || true
  rm -f "${TEST_POD_FILE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

for cmd in bash kubectl grep sed; do
  require_cmd "${cmd}"
done

log "Checking node readiness"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=condition=Ready node --all --timeout='${WAIT_TIMEOUT}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get nodes -o wide"

log "Checking GPU allocatable"
GPU_ALLOC_LINE="$(kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" alloc="}{.status.allocatable.nvidia\.com/gpu}{" cap="}{.status.capacity.nvidia\.com/gpu}{"\\n"}{end}')"
log "${GPU_ALLOC_LINE}"
if ! grep -Eq 'alloc=[1-9][0-9]*' <<<"${GPU_ALLOC_LINE}"; then
  echo "GPU is not allocatable in k3s node(s)." >&2
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system get pods -o wide | grep -Ei 'nvidia|device-plugin|gpu' || true"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system logs daemonset/nvidia-device-plugin-daemonset --all-containers=true --tail=200 || true"
  exit 1
fi

cat >"${TEST_POD_FILE}" <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${TEST_POD_NAME}
spec:
  restartPolicy: Never
  runtimeClassName: ${RUNTIME_CLASS_NAME}
  containers:
  - name: gpu
    image: ${TEST_POD_IMAGE}
    imagePullPolicy: Never
    command: ["bash", "-lc", "nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
YAML

log "Applying smoke-test pod"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' delete pod '${TEST_POD_NAME}' --ignore-not-found=true"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '${TEST_POD_FILE}'"

if ! kubectl --kubeconfig "${K3S_KUBECONFIG}" wait --for=jsonpath='{.status.phase}'=Succeeded "pod/${TEST_POD_NAME}" --timeout="${WAIT_TIMEOUT}"; then
  log "Pod did not reach Succeeded. Collecting diagnostics."
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get pod '${TEST_POD_NAME}' -o wide"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' describe pod '${TEST_POD_NAME}' | sed -n '1,260p'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' logs '${TEST_POD_NAME}' --tail=200 || true"
  exit 1
fi

run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get pod '${TEST_POD_NAME}' -o wide"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' logs '${TEST_POD_NAME}' --tail=200"

log "SUCCESS: nvidia-smi worked in k3s pod with image ${TEST_POD_IMAGE}."
