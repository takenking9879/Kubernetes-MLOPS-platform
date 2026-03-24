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

# Python dependencies
COPY requirements_gpu.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm -rf /tmp/* && pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Return to default ray user
USER ray