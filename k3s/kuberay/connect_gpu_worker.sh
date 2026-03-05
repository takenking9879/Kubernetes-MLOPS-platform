#!/bin/bash

# Configuration
IMAGE_NAME="ray-cluster-gpu:2.53.0" # Ensure this image is available or built
HEAD_NAMESPACE="ray"

# 1. Find Head Pod IP
echo "[1/3] Finding Ray Head Pod IP..."
HEAD_IP=$(kubectl get pod -n $HEAD_NAMESPACE -l ray.io/node-type=head -o jsonpath='{.items[0].status.podIP}')

if [ -z "$HEAD_IP" ]; then
    echo "Error: Could not find Ray Head Pod. Is the RayJob running?"
    exit 1
fi

echo "Found Head IP: $HEAD_IP"

# 2. Extract Host IP for Ray worker connection
HOST_IP=$(ip route get $HEAD_IP | awk '{print $7; exit}')
echo "[2/3] Host IP (for node-ip-address): $HOST_IP"

# 3. Launch Docker GPU Worker
echo "[3/3] Launching Docker container with GPU worker..."
docker run --rm -it \
    --gpus all \
    --network host \
    --shm-size=5gb \
    $IMAGE_NAME \
    ray start \
        --address="$HEAD_IP:6379" \
        --node-ip-address="$HOST_IP" \
        --num-gpus=1 \
        --num-cpus=2 \
        --min-worker-port=20000 \
        --max-worker-port=20020 \
        --block
