FROM rayproject/ray:2.53.0-py311

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Copy GPU requirements (PyTorch CUDA 12.8 + XGBoost GPU)
COPY k3s/kuberay/gpu/requirements_gpu.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --system \
    --no-cache \
    -r /tmp/requirements.txt

# Install Tailscale — required for RunPod workers to join the Ray cluster
# via a WireGuard mesh VPN when the K8s cluster is not publicly reachable.
USER root
RUN curl -fsSL https://tailscale.com/install.sh | sh

# Copy the worker startup script
COPY k3s/runpod/runpod-worker-start.sh /usr/local/bin/runpod-worker-start.sh
RUN chmod +x /usr/local/bin/runpod-worker-start.sh

# Clean up
RUN rm -rf /tmp/* /var/tmp/*

USER ray

ENTRYPOINT ["/usr/local/bin/runpod-worker-start.sh"]
