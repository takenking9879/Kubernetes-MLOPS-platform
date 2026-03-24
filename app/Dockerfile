############################
# Stage 1: Python + Backend deps
############################
FROM python:3.11.14-slim-bookworm AS python-builder

ENV DEBIAN_FRONTEND=noninteractive

# ========================
# 1. Dependencias base
# ========================
RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-distutils \
    curl \
    git \
    build-essential \
    ca-certificates \
    gnupg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ========================
# 2. Copiar y instalar backend deps
# ========================
WORKDIR /deps
COPY backend/requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/usr/local -r requirements.txt \
    && pip install --prefix=/usr/local vastai-sdk==0.5.1 --no-deps

############################
# Stage 2: Spark + Node.js + frontend + Python runtime
############################
FROM spark:4.0.1

USER root
ENV DEBIAN_FRONTEND=noninteractive

# ========================
# 1. Instalar Python y Node.js
# ========================
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-distutils curl \
    && curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python --version \
    && node --version \
    && npm --version \
    && rm -rf /var/lib/apt/lists/*

# ========================
# 2. Copiar paquetes Python desde el builder
# ========================
COPY --from=python-builder /usr/local /usr/local

# ========================
# 3. Crear workspace y copiar código
# ========================
WORKDIR /workspace
RUN mkdir -p backend frontend

COPY backend/ backend/
COPY frontend/ frontend/

# ========================
# 4. Instalar frontend deps
# ========================
RUN if [ -f frontend/package.json ]; then \
        cd frontend && npm install ; \
    else \
        echo "No package.json found, skipping frontend deps"; \
    fi

# -------------------------------
# Aquí agregamos los JARs de S3
# -------------------------------
# Crear carpeta para jars si no existe
RUN mkdir -p /opt/spark/jars

# Descargar hadoop-aws, aws-java-sdk y commons-pool2
RUN wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar \
    # && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.797/aws-java-sdk-bundle-1.12.797.jar \
    \
    # Añadidas dependencias SDK aws
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/s3/2.41.8/s3-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/sdk-core/2.41.8/sdk-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/annotations/2.41.8/annotations-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/utils/2.41.8/utils-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-core/2.41.8/aws-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/auth/2.41.8/auth-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/regions/2.41.8/regions-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-client-spi/2.41.8/http-client-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/netty-nio-client/2.41.8/netty-nio-client-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/protocol-core/2.41.8/protocol-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/profiles/2.41.8/profiles-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/endpoints-spi/2.41.8/endpoints-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/identity-spi/2.41.8/identity-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/s3-transfer-manager/2.41.8/s3-transfer-manager-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/metrics-spi/2.41.8/metrics-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/apache-client/2.41.8/apache-client-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/retries-spi/2.41.8/retries-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/retries/2.41.8/retries-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-auth-spi/2.41.8/http-auth-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-auth/2.41.8/http-auth-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/http-auth-aws/2.41.8/http-auth-aws-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/reactivestreams/reactive-streams/1.0.4/reactive-streams-1.0.4.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-xml-protocol/2.41.8/aws-xml-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-query-protocol/2.41.8/aws-query-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/checksums-spi/2.41.8/checksums-spi-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/checksums/2.41.8/checksums-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-json-protocol/2.41.8/aws-json-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/aws-cbor-protocol/2.41.8/aws-cbor-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/smithy-rpcv2-protocol/2.41.8/smithy-rpcv2-protocol-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/json-utils/2.41.8/json-utils-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/third-party-jackson-core/2.41.8/third-party-jackson-core-2.41.8.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/software/amazon/awssdk/arns/2.41.8/arns-2.41.8.jar

# Iceberg runtime y extensiones para Spark 4.x
RUN wget -P /opt/spark/jars/ https://repo.maven.apache.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.0_2.13/1.10.1/iceberg-spark-runtime-4.0_2.13-1.10.1.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.10.1/iceberg-aws-bundle-1.10.1.jar \
    && wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-extensions-4.0_2.13/1.10.1/iceberg-spark-extensions-4.0_2.13-1.10.1.jar

RUN chown -R spark:spark /workspace && chmod -R 775 /workspace

USER spark
