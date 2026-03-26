# SkyPilot runner image — runs sky operations from Airflow KubernetesPodOperator.
#
# Why NOT berkeleyskypilot/skypilot:0.12.0:
#   The official image is ~2 GB (GCloud SDK 400 MB, Azure CLI 600 MB, kubectl, etc.).
#   SkyPilot only needs the Python SDKs (boto3, runpod, vastai) — no cloud CLIs.
#   The CLIs are for interactive setup (aws configure, runpod config), not K8s pods.
#
# Build from REPO ROOT:
#   docker build -f k3s/sky/Dockerfile -t takenking9879/sky-runner:0.12.0 .
#   docker push takenking9879/sky-runner:0.12.0
#
# Cloud credentials flow (all from env-secret, no extra secrets needed):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY → boto3 reads natively
#   RUNPOD_API_KEY  → entrypoint writes ~/.runpod/config.toml
#   VASTAI_API_KEY  → entrypoint writes ~/.config/vastai/vast_api_key

FROM python:3.11.14-slim-bookworm

WORKDIR /app

# System deps: git + ssh (SkyPilot provisioning), rsync (file sync to VMs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    rsync \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY k3s/sky/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy runner script and entrypoint
COPY k3s/sky/sky_runner.py /app/sky_runner.py
COPY k3s/sky/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Bake SkyPilot task YAMLs (rebuild image when YAMLs change)
COPY k3s/sky/*.yaml /app/k3s/sky/

# Copy GPU catalog/selector for dynamic resource selection
# (gracefully falls back to static YAML any_of if import fails)
COPY src/ /app/src/

ENV SKY_YAML_DIR=/app/k3s/sky

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "/app/sky_runner.py"]
