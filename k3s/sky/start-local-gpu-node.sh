#!/bin/bash
# Start (or restart) the local GPU SSH node container for SkyPilot.
#
# Run on the WSL host BEFORE registering the SSH Node Pool in the SkyPilot API
# server pod. Only needs to be run once; the container uses --restart unless-stopped.
#
# Usage:
#   bash k3s/sky/start-local-gpu-node.sh
#
# Environment overrides:
#   LOCAL_GPU_SSH_PORT  — host port for SSH (default: 2222)
#   LOCAL_GPU_IMAGE     — Docker image to use (default: takenking9879/ray-train-ssh:2.53.0)
set -euo pipefail

IMAGE="${LOCAL_GPU_IMAGE:-takenking9879/ray-train-ssh:2.53.0}"
CONTAINER_NAME="sky-local-gpu"
SSH_PORT="${LOCAL_GPU_SSH_PORT:-2222}"
SSH_KEY_PATH="${HOME}/.sky/ssh_keys/local-gpu"

echo "[local-gpu] Image: ${IMAGE}"
echo "[local-gpu] SSH port: ${SSH_PORT}"
echo "[local-gpu] SSH key: ${SSH_KEY_PATH}"

# Generate SSH key pair if not present
if [ ! -f "${SSH_KEY_PATH}" ]; then
    mkdir -p "$(dirname "${SSH_KEY_PATH}")"
    ssh-keygen -t ed25519 -f "${SSH_KEY_PATH}" -N ""
    echo "[local-gpu] Generated new SSH key pair at ${SSH_KEY_PATH}"
fi
AUTHORIZED_KEY="$(cat "${SSH_KEY_PATH}.pub")"

# Stop and remove any existing container with the same name
if docker inspect "${CONTAINER_NAME}" &>/dev/null; then
    echo "[local-gpu] Removing existing container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}"
fi

# Start the container with GPU access and SSH exposed
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${SSH_PORT}:22" \
    -e "SSH_AUTHORIZED_KEY=${AUTHORIZED_KEY}" \
    --restart unless-stopped \
    "${IMAGE}"

echo ""
echo "[local-gpu] Container started."
echo ""
echo "  Verify GPU:"
echo "    docker exec ${CONTAINER_NAME} nvidia-smi"
echo ""
echo "  Verify SSH from WSL host:"
echo "    ssh -p ${SSH_PORT} -i ${SSH_KEY_PATH} -o StrictHostKeyChecking=no root@127.0.0.1 nvidia-smi"
echo ""
echo "  Verify SSH from K8s pod (SkyPilot API server):"
echo "    kubectl exec -it -n skypilot deployment/skypilot-api-server -- \\"
echo "      ssh -p ${SSH_PORT} -i ~/.sky/ssh_keys/local-gpu root@host.docker.internal nvidia-smi"
echo ""
echo "  Next step: register SSH Node Pool in SkyPilot API server pod."
echo "  See k3s/sky/ray-gpu-training-local.yaml header for full setup instructions."
