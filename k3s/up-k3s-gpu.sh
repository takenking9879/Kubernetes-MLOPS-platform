#!/usr/bin/env bash
set -euo pipefail

# Bring up local k3s using containerd, enable NVIDIA runtime, and deploy GPU stack.
# This script ASSUMES your training image is already imported in k3s containerd.
#
# Usage:
#   sudo bash k3s/up-k3s-gpu.sh
#
# Optional env vars:
#   START_LOCAL_DOCKER=0
#   INSTALL_K3S_VERSION=v1.34.1+k3s1
#   K3S_HTTPS_PORT=16443
#   WAIT_TIMEOUT=240s
#   GPU_ALLOC_WAIT_TIMEOUT=240
#   DEVICE_PLUGIN_VERSION=v0.17.1
#   ENSURE_NVIDIA_TOOLKIT=1
#   VERIFY_LOCAL_IMAGE=1
#   LOCAL_IMAGE=takenking9879/ray-train:2.53.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${SCRIPT_DIR}/bin:${PATH}"

START_LOCAL_DOCKER="${START_LOCAL_DOCKER:-0}"
INSTALL_K3S_VERSION="${INSTALL_K3S_VERSION:-v1.34.1+k3s1}"
K3S_HTTPS_PORT="${K3S_HTTPS_PORT:-16443}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-240s}"
GPU_ALLOC_WAIT_TIMEOUT="${GPU_ALLOC_WAIT_TIMEOUT:-240}"
DOCKER_READY_TIMEOUT="${DOCKER_READY_TIMEOUT:-30}"
DEVICE_PLUGIN_VERSION="${DEVICE_PLUGIN_VERSION:-v0.17.1}"
DEVICE_PLUGIN_MANIFEST="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${DEVICE_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml"

ENSURE_NVIDIA_TOOLKIT="${ENSURE_NVIDIA_TOOLKIT:-1}"
VERIFY_LOCAL_IMAGE="${VERIFY_LOCAL_IMAGE:-1}"
LOCAL_IMAGE="${LOCAL_IMAGE:-takenking9879/ray-train:2.53.0}"

K3S_KUBECONFIG="${K3S_KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
K3S_SERVICE_ENV_FILE="/etc/systemd/system/k3s.service.env"
K3S_SERVICE_FILE="/etc/systemd/system/k3s.service"
RUNTIME_CLASS_FILE="${SCRIPT_DIR}/kuberay/nvidia-runtime-class.yaml"

log() {
  echo "[up-k3s-gpu] $*"
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

unit_exists() {
  local unit="$1"
  systemctl list-unit-files "${unit}" --no-legend 2>/dev/null | awk '{print $1}' | grep -Fxq "${unit}"
}

is_docker_ready() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

wait_for_docker_ready() {
  local elapsed=0
  while (( elapsed < DOCKER_READY_TIMEOUT )); do
    if is_docker_ready; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
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

sanitize_wsl_docker_host_mount() {
  if grep -q ' /Docker/host ' /proc/mounts 2>/dev/null; then
    log "Detected /Docker/host mount with spaces in source path; unmounting to avoid kubelet mount parser failures."
    run "umount /Docker/host || true"
  fi
}

timeout_to_seconds() {
  local raw="$1"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    echo "${raw}"
    return
  fi
  if [[ "${raw}" =~ ^([0-9]+)s$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "${raw}" =~ ^([0-9]+)m$ ]]; then
    echo "$(( BASH_REMATCH[1] * 60 ))"
    return
  fi
  echo "240"
}

wait_for_kube_api() {
  local timeout_seconds="$1"
  local elapsed=0
  while (( elapsed < timeout_seconds )); do
    if kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

wait_for_gpu_allocatable() {
  local elapsed=0
  while (( elapsed < GPU_ALLOC_WAIT_TIMEOUT )); do
    local alloc_line
    alloc_line="$(kubectl --kubeconfig "${K3S_KUBECONFIG}" get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" alloc="}{.status.allocatable.nvidia\.com/gpu}{" cap="}{.status.capacity.nvidia\.com/gpu}{"\\n"}{end}' 2>/dev/null || true)"
    if [[ -n "${alloc_line}" ]]; then
      log "${alloc_line}"
      if grep -Eq 'alloc=[1-9][0-9]*' <<<"${alloc_line}"; then
        return 0
      fi
    else
      log "k3s API temporarily unavailable while waiting for GPU allocatable."
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done
  return 1
}

ensure_k3s_containerd_mode() {
  local k3s_runtime_flag=""
  if [[ "${START_LOCAL_DOCKER}" == "1" ]]; then
    k3s_runtime_flag="--docker"
  fi

  if ! unit_exists "k3s.service"; then
    if pgrep -x k3s >/dev/null 2>&1; then
      log "k3s process is already running and k3s.service unit is not discoverable. Skipping installation."
      return
    fi
    if command -v k3s >/dev/null 2>&1 && [[ -f "${K3S_KUBECONFIG}" ]]; then
      log "k3s binary/kubeconfig exist but k3s.service unit is not discoverable. Skipping installation."
      return
    fi

    if command -v ss >/dev/null 2>&1 && ss -lnt "( sport = :${K3S_HTTPS_PORT} )" | grep -q ":${K3S_HTTPS_PORT}"; then
      echo "Port ${K3S_HTTPS_PORT} is already in use. Set K3S_HTTPS_PORT to a free port." >&2
      exit 1
    fi

    local install_exec
    install_exec="server ${k3s_runtime_flag} --disable traefik --write-kubeconfig-mode 644 --https-listen-port ${K3S_HTTPS_PORT}"
    log "k3s not found. Installing k3s ${INSTALL_K3S_VERSION}"
    run "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='${INSTALL_K3S_VERSION}' INSTALL_K3S_EXEC='${install_exec}' sh -"
    return
  fi

  local runtime_mismatch=0
  local currently_docker=0
  if k3s_uses_docker_runtime; then
    currently_docker=1
  fi

  if [[ "${START_LOCAL_DOCKER}" == "1" ]]; then
    if [[ "${currently_docker}" == "0" ]]; then
      runtime_mismatch=1
    fi
  else
    if [[ "${currently_docker}" == "1" ]]; then
      runtime_mismatch=1
    fi
  fi

  if [[ "${runtime_mismatch}" == "1" ]]; then
    local desired="containerd"
    if [[ "${START_LOCAL_DOCKER}" == "1" ]]; then
      desired="docker"
    fi
    local install_exec
    install_exec="server ${k3s_runtime_flag} --disable traefik --write-kubeconfig-mode 644 --https-listen-port ${K3S_HTTPS_PORT}"
    log "k3s runtime mismatch. Reconfiguring k3s service to use ${desired}."
    run "systemctl stop k3s || true"
    run "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='${INSTALL_K3S_VERSION}' INSTALL_K3S_EXEC='${install_exec}' sh -"
  fi
}

ensure_nvidia_runtime_for_k3s() {
  if [[ "${ENSURE_NVIDIA_TOOLKIT}" != "1" ]]; then
    log "Skipping NVIDIA toolkit installation/configuration (ENSURE_NVIDIA_TOOLKIT=${ENSURE_NVIDIA_TOOLKIT})."
    return
  fi

  if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    log "Installing nvidia-container-toolkit"
    run "apt-get update"
    run "apt-get install -y nvidia-container-toolkit"
  fi

  # k3s auto-detects /usr/bin/nvidia-container-runtime; no image import or host ctr changes here.
  log "Restarting k3s to apply NVIDIA runtime detection"
  sanitize_wsl_docker_host_mount
  run "systemctl restart k3s"
  run "systemctl is-active k3s"

  local timeout_seconds
  timeout_seconds="$(timeout_to_seconds "${WAIT_TIMEOUT}")"
  if ! wait_for_kube_api "${timeout_seconds}"; then
    echo "k3s API did not become reachable after restart within ${WAIT_TIMEOUT}." >&2
    exit 1
  fi

  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=condition=Ready node --all --timeout='${WAIT_TIMEOUT}'"
}

deploy_gpu_stack() {
  log "Applying RuntimeClass nvidia"
  if [[ -f "${RUNTIME_CLASS_FILE}" ]]; then
    run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '${RUNTIME_CLASS_FILE}'"
  else
    run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f - <<'YAML'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
YAML"
  fi

  log "Deploying NVIDIA device plugin ${DEVICE_PLUGIN_VERSION}"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' apply -f '${DEVICE_PLUGIN_MANIFEST}'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system patch daemonset nvidia-device-plugin-daemonset --type merge -p '{\"spec\":{\"template\":{\"spec\":{\"runtimeClassName\":\"nvidia\"}}}}'"
  run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout='${WAIT_TIMEOUT}'"

  log "Waiting for GPU allocatable in node status"
  if ! wait_for_gpu_allocatable; then
    log "GPU allocatable did not appear in time. Collecting diagnostics."
    run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get nodes -o wide"
    run "kubectl --kubeconfig '${K3S_KUBECONFIG}' describe node $(kubectl --kubeconfig '${K3S_KUBECONFIG}' get nodes -o jsonpath='{.items[0].metadata.name}') | sed -n '1,260p'"
    run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system get pods -o wide | grep -Ei 'nvidia|device-plugin|gpu' || true"
    run "kubectl --kubeconfig '${K3S_KUBECONFIG}' -n kube-system logs daemonset/nvidia-device-plugin-daemonset --all-containers=true --tail=300 || true"
    exit 1
  fi
}

setup_kubectl_and_kubeconfig() {
  if ! command -v kubectl >/dev/null 2>&1; then
    log "No kubectl found — installing wrapper to /usr/local/bin/kubectl"
    cp "${SCRIPT_DIR}/bin/kubectl" /usr/local/bin/kubectl
    chmod +x /usr/local/bin/kubectl
  fi

  # k3s names its context/cluster/user "default" — rename to "k3s" to avoid
  # clobbering docker-desktop or kubeadm contexts that also use "default"
  local tmp_cfg
  tmp_cfg="$(mktemp)"
  sed -e 's/\(cluster:\|user:\|name:\) default$/\1 k3s/g' \
      -e 's/current-context: default/current-context: k3s/' \
      "${K3S_KUBECONFIG}" > "${tmp_cfg}"

  _merge_and_switch_context "/root/.kube" "root" "${tmp_cfg}"

  if [[ -n "${SUDO_USER:-}" ]]; then
    local user_home
    user_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    if [[ -n "${user_home}" ]]; then
      _merge_and_switch_context "${user_home}/.kube" "${SUDO_USER}" "${tmp_cfg}"
    fi
  fi

  rm -f "${tmp_cfg}"

  # Make k3s kubectl respect ~/.kube/config instead of /etc/rancher/k3s/k3s.yaml
  if [[ -n "${SUDO_USER:-}" ]]; then
    local user_home
    user_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    local kubeconfig_line='export KUBECONFIG="$HOME/.kube/config"'
    for rc in "${user_home}/.bashrc" "${user_home}/.zshrc"; do
      if [[ -f "${rc}" ]] && ! grep -q 'KUBECONFIG' "${rc}"; then
        echo "${kubeconfig_line}" >> "${rc}"
        log "Added KUBECONFIG export to ${rc}"
      fi
    done
  fi
}

_merge_and_switch_context() {
  local kube_dir="$1" owner="$2" src="$3"
  local dest="${kube_dir}/config"
  mkdir -p "${kube_dir}"
  if [[ -f "${dest}" ]]; then
    KUBECONFIG="${dest}:${src}" kubectl config view --flatten > "${dest}.tmp"
    mv "${dest}.tmp" "${dest}"
  else
    cp "${src}" "${dest}"
  fi
  chmod 600 "${dest}"
  [[ "${owner}" != "root" ]] && chown "${owner}:${owner}" "${kube_dir}" "${dest}"
  # Remove legacy k3s "default" context that existed before the rename
  kubectl --kubeconfig "${dest}" config delete-context default >/dev/null 2>&1 || true
  kubectl --kubeconfig "${dest}" config delete-cluster default >/dev/null 2>&1 || true
  kubectl --kubeconfig "${dest}" config unset "users.default" >/dev/null 2>&1 || true
  kubectl --kubeconfig "${dest}" config use-context k3s
  log "kubeconfig merged, current-context=k3s for ${owner}"
}

verify_local_image_in_k3s_containerd() {
  if [[ "${VERIFY_LOCAL_IMAGE}" != "1" ]]; then
    log "Skipping local image verification (VERIFY_LOCAL_IMAGE=${VERIFY_LOCAL_IMAGE})."
    return
  fi

  local escaped
  escaped="${LOCAL_IMAGE//\//\\/}"

  if run "k3s ctr -n k8s.io images ls | awk '{print \$1}' | grep -Ex '(docker\.io/${escaped}|${escaped})'" >/dev/null 2>&1; then
    log "Local image already present in k3s containerd: ${LOCAL_IMAGE}"
    return
  fi

  echo "Required image not found in k3s containerd: ${LOCAL_IMAGE}" >&2
  echo "Manual import example:" >&2
  echo "  docker save -o /tmp/ray-train-2.53.0.tar ${LOCAL_IMAGE}" >&2
  echo "  sudo k3s ctr -n k8s.io images import /tmp/ray-train-2.53.0.tar" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo/root." >&2
  exit 1
fi

for cmd in bash curl systemctl awk grep sed; do
  require_cmd "${cmd}"
done

if [[ "${START_LOCAL_DOCKER}" == "1" ]]; then
  if unit_exists "docker.service"; then
    log "Starting docker.service because START_LOCAL_DOCKER=1"
    run "systemctl start docker"
    run "systemctl start docker.socket || true"
  fi
  if ! wait_for_docker_ready; then
    echo "START_LOCAL_DOCKER=1 but docker daemon is not ready." >&2
    exit 1
  fi
fi

ensure_k3s_containerd_mode
sanitize_wsl_docker_host_mount

log "Starting k3s (boot auto-start disabled; use up-k3s-gpu.sh to start manually)"
run "systemctl disable k3s 2>/dev/null || true"
run "systemctl start k3s"
run "systemctl is-active k3s"
api_timeout_seconds="$(timeout_to_seconds "${WAIT_TIMEOUT}")"
if ! wait_for_kube_api "${api_timeout_seconds}"; then
  echo "k3s API did not become reachable after service start within ${WAIT_TIMEOUT}." >&2
  exit 1
fi
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' wait --for=condition=Ready node --all --timeout='${WAIT_TIMEOUT}'"
run "kubectl --kubeconfig '${K3S_KUBECONFIG}' get nodes -o wide"

ensure_nvidia_runtime_for_k3s
deploy_gpu_stack
verify_local_image_in_k3s_containerd
setup_kubectl_and_kubeconfig

# k3s on WSL2 only: pre-seed Windows hosts with current WSL2 IP so *.localhost
# resolves correctly after deploy. Not needed with Docker Desktop / kubeadm.
bash "${SCRIPT_DIR}/update-win-hosts.sh" || true

log "Done. k3s is up — 'kubectl get pods -A' should work in your terminal now."
