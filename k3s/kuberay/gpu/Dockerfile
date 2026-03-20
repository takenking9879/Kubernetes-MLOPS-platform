FROM rayproject/ray:2.53.0-py311

USER root

# Install minimal tools required
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python installer)
RUN pip install --no-cache-dir uv

# Install Tailscale using official script
RUN curl -fsSL https://tailscale.com/install.sh | sh

# Python dependencies
COPY requirements_gpu.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm -rf /tmp/*

# Return to default ray user
USER ray