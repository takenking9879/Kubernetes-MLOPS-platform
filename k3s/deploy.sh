#!/usr/bin/env bash
set -euo pipefail

# Correr ./k3s/deploy.sh para desplegar la plataforma completa en k3s.

# ============================================================
# FLAGS (EDITA AQUÍ)
# ============================================================
ENABLE_REPO_DOWNLOAD=false
ENABLE_KAFKA=true
ENABLE_RAY=true
ENABLE_MLFLOW=false
ENABLE_SPARK=true

# ============================================================
# TIMEOUTS
# ============================================================
KAFKA_OPERATOR_TIMEOUT="200s"
KAFKA_CLUSTER_TIMEOUT="300s"
ACCESS_OPERATOR_TIMEOUT="200s"
POSTGRES_TIMEOUT="300s"
INGRESS_CONTROLLER_TIMEOUT="200s"
INGRESS_ADMISSION_TIMEOUT="120s"

# ============================================================
# NAMESPACES
# ============================================================
NS_KAFKA="kafka"
NS_RAY="ray"
NS_SPARK="spark"
ENV_FILE=".env"

# ============================================================
# LOGGING
# ============================================================
sep() { echo; echo "============================================================"; echo; }
info() { echo "👉 $*"; }
ok() { echo "✅ $*"; }

# ============================================================
# HELM REPOS
# ============================================================
if [ "${ENABLE_REPO_DOWNLOAD}" = true ]; then
  sep
  info "Adding Helm repos"
  helm repo add strimzi https://strimzi.io/charts
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
  helm repo add spark-kubernetes-operator https://apache.github.io/spark-kubernetes-operator/
  helm repo update
  ok "Helm repos ready"
fi

# ============================================================
# NAMESPACES + SECRETS
# ============================================================
sep
info "Creating namespaces"
kubectl create namespace kafka || true
kubectl create namespace ray || true
kubectl create namespace spark || true

info "Creating secrets from .env"
kubectl create secret generic kafka-secret -n kafka --from-env-file=.env || true
kubectl create secret generic env-secret -n ray --from-env-file=.env || true
kubectl create secret generic env-secret -n spark --from-env-file=.env || true
ok "Namespaces and secrets ready"

# ============================================================
# KAFKA (STRICTLY SEQUENTIAL, REPLICA-SAFE)
# ============================================================
deploy_kafka() {
  sep
  info "Deploying Kafka (Strimzi)"

  # ------------------------------------------------------------
  # 1. Install Strimzi operator
  # ------------------------------------------------------------
  helm install my-strimzi-kafka-operator strimzi/strimzi-kafka-operator \
    --version 0.49.1 \
    -n kafka \
    --create-namespace

  info "Waiting for Strimzi cluster operator"
  kubectl wait \
    deployment/strimzi-cluster-operator \
    -n kafka \
    --for=condition=Available \
    --timeout="${KAFKA_OPERATOR_TIMEOUT}"

  # ------------------------------------------------------------
  # 2. Apply Kafka cluster (NodePool + Kafka CR)
  # ------------------------------------------------------------
  info "Applying Kafka cluster CR"
  kubectl apply -f k3s/kafka/kafka-cluster.yaml

  # ------------------------------------------------------------
  # 3. Wait for Kafka CR to be Ready (Strimzi-level readiness)
  # ------------------------------------------------------------
  info "Waiting for Kafka CR to be Ready"
  kubectl wait \
    kafka/my-kafka-cluster \
    -n kafka \
    --for=condition=Ready \
    --timeout="${KAFKA_CLUSTER_TIMEOUT}"

  # ------------------------------------------------------------
  # 4. Wait for ALL broker/controller pods (replica-safe)
  # ------------------------------------------------------------
  sleep 15 # small buffer time
  info "Waiting for all my-kafka-cluster-dual-role-x pods to be Ready"

  # ------------------------------------------------------------
  # 5. Wait for Entity Operator (topics + users)
  # ------------------------------------------------------------
  info "Waiting for Entity Operator"
  kubectl wait \
    deployment/my-kafka-cluster-entity-operator \
    -n kafka \
    --for=condition=Available \
    --timeout="${KAFKA_CLUSTER_TIMEOUT}"

  ok "Kafka cluster fully ready"

  # ------------------------------------------------------------
  # 6. Apply topics and users
  # ------------------------------------------------------------
  info "Applying topics and users"
  kubectl apply -f k3s/kafka/kafka-topics.yaml
  kubectl apply -f k3s/kafka/kafka-user.yaml

  # ------------------------------------------------------------
  # 7. Install Access Operator
  # ------------------------------------------------------------
  info "Installing access-operator"
  kubectl create -f k3s/kafka/install/access-operator || true

  kubectl wait \
    deployment/strimzi-access-operator \
    -n strimzi-access-operator \
    --for=condition=Available \
    --timeout="${ACCESS_OPERATOR_TIMEOUT}"

  # ------------------------------------------------------------
  # 8. Apply access + UI
  # ------------------------------------------------------------
  kubectl apply -f k3s/kafka/kafka-access.yaml
  kubectl apply -f k3s/kafka/kafka-ui.yaml

  ok "Kafka fully deployed 🚀"
}

# ============================================================
# RAY (PARALLEL)
# ============================================================
deploy_ray() {
  sep
  info "Deploying KubeRay Operator"
  helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.5.1 -n ray
  ok "KubeRay operator installed"
}

# ============================================================
# MLFLOW (PARALLEL)
# ============================================================
deploy_mlflow() {
  sep
  info "Deploying Postgres"
  helm install postgres bitnami/postgresql \
    -n ray -f k3s/mlflow/postgres_values.yaml

  kubectl wait \
    --for=condition=Ready pod \
    -l app.kubernetes.io/instance=postgres \
    -n ray --timeout="${POSTGRES_TIMEOUT}"

  info "Deploying MLflow"
  helm install my-mlflow community-charts/mlflow \
    -n ray -f k3s/mlflow/mlflow_values.yaml

  ok "MLflow deployed"
}

# ============================================================
# SPARK (PARALLEL)
# ============================================================
deploy_spark() {
  sep
  info "Deploying Spark Operator"
  helm install my-spark-kubernetes-operator \
    spark-kubernetes-operator/spark-kubernetes-operator \
    --version 1.4.0 \
    -n spark -f k3s/spark/values.yaml
  ok "Spark operator deployed"
}

# ============================================================
# INGRESS (ABSOLUTELY LAST)
# ============================================================
deploy_ingress() {
  sep
  info "Deploying Ingress NGINX controller"

  helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --create-namespace \
    -f k3s/ingress-controller/ingress-values.yaml

  info "Waiting for ingress-nginx controller"
  kubectl wait \
    deployment/ingress-nginx-controller \
    -n ingress-nginx \
    --for=condition=Available \
    --timeout="${INGRESS_CONTROLLER_TIMEOUT}"

  ok "Ingress NGINX ready"

  info "Deploying platform ingress (FINAL STEP)"
  helm upgrade --install platform-ingress ./k3s/ingress \
    -f k3s/ingress/values.yaml

  ok "Platform ingress deployed 🚀"
}

# ============================================================
# EXECUTION FLOW
# ============================================================

if [ "${ENABLE_KAFKA}" = true ]; then
  deploy_kafka
fi

sep
info "Launching PARALLEL workloads (Ray / MLflow / Spark)"

PIDS=()

if [ "${ENABLE_RAY}" = true ]; then
  deploy_ray & PIDS+=($!)
fi

if [ "${ENABLE_MLFLOW}" = true ]; then
  deploy_mlflow & PIDS+=($!)
fi

if [ "${ENABLE_SPARK}" = true ]; then
  deploy_spark & PIDS+=($!)
fi

info "Waiting for parallel jobs to finish"
wait "${PIDS[@]}"
ok "Parallel workloads completed"

deploy_ingress
sep
ok "DEPLOYMENT FINISHED SUCCESSFULLY"