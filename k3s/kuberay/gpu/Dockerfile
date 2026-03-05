FROM rayproject/ray:2.53.0-py311

# 1️⃣ Instalar uv (binario precompilado, súper rápido)
RUN pip install --no-cache-dir uv

# 2️⃣ Copiar requirements primero (mejor cache de Docker)
COPY requirements_gpu.txt /tmp/requirements.txt

# 3️⃣ Instalar deps con uv (mucho más rápido que pip)
RUN uv pip install \
    --system \
    --no-cache \
    -r /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --system \
    --no-cache \
    -r /tmp/requirements.txt && \
    # Limpieza agresiva de archivos que no necesitas en el runtime
    sudo rm -rf /tmp/* /var/tmp/*