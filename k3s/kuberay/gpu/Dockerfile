FROM rayproject/ray:2.53.0-py311

USER root

# Install packages required by SkyPilot's apt-ssh-setup (pre-installing avoids
# slow runtime apt downloads inside K8s pods) + minimal tools
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        rsync \
        wget \
        netcat-openbsd \
        gcc \
        patch \
        pciutils \
        fuse3 \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python installer)
RUN pip install --no-cache-dir uv

# Python dependencies
COPY requirements_gpu.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm -rf /tmp/*

# Return to default ray user
USER ray