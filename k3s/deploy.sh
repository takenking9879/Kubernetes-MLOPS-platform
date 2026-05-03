#!/usr/bin/env bash
set -euo pipefail

# Correr ./k3s/deploy.sh para desplegar la plataforma completa en k3s.

# ============================================================
# FLAGS (EDITA AQUÍ)
# ============================================================
ENABLE_REPO_DOWNLOAD=false
ENABLE_KAFKA=false
ENABLE_RAY=false
ENABLE_MLFLOW=true
ENABLE_SPARK=true
ENABLE_APP=true
ENABLE_MONITORING=true
ENABLE_AIRFLOW=true
ENABLE_SKYPILOT=true
ENABLE_INGRESS=true
ENABLE_CERT_MANAGER=false

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
NS_MONITORING="monitoring"
NS_APPS="apps"

# ============================================================
# LOGGING
# ============================================================
sep() { echo; echo "============================================================"; echo; }
info() { echo "👉 $*"; }
ok() { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }

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
  helm repo add skypilot https://helm.skypilot.co/
  helm repo update
  ok "Helm repos ready"
fi

# ============================================================
# GPU NODE LABELING (SkyPilot)
# ============================================================
_map_nvidia_name_to_skypilot() {
  local raw
  raw=$(echo "$1" | tr '[:upper:]' '[:lower:]' \
    | sed 's/nvidia //g; s/geforce //g; s/ laptop gpu//g; s/laptop gpu//g; s/-laptop-gpu//g' \
    | sed 's/nvidia-//g; s/geforce-//g' \
    | xargs)
  case "$raw" in
    "rtx 5090"|"rtx5090"|"rtx-5090")           echo "rtx5090" ;;
    "rtx 4090"|"rtx4090"|"rtx-4090")           echo "rtx4090" ;;
    "rtx 4080"|"rtx4080"|"rtx-4080")           echo "rtx4080" ;;
    "rtx 4070 ti super"|"rtx4070tisuper")       echo "rtx4070ti" ;;
    "rtx 4070 ti"|"rtx4070ti"|"rtx-4070-ti")   echo "rtx4070ti" ;;
    "rtx 4070 super"|"rtx4070super")            echo "rtx4070" ;;
    "rtx 4070"|"rtx4070"|"rtx-4070")           echo "rtx4070" ;;
    "rtx 4060"|"rtx4060"|"rtx-4060")           echo "rtx4060" ;;
    "rtx 3090 ti"|"rtx3090ti"|"rtx-3090-ti")   echo "rtx3090" ;;
    "rtx 3090"|"rtx3090"|"rtx-3090")           echo "rtx3090" ;;
    "rtx 3080 ti"|"rtx3080ti"|"rtx-3080-ti")   echo "rtx3080" ;;
    "rtx 3080"|"rtx3080"|"rtx-3080")           echo "rtx3080" ;;
    *"a100"*"80gb"*"sxm"*)  echo "a100-80gb-sxm" ;;
    *"a100"*"80gb"*)         echo "a100-80gb" ;;
    *"a100"*)                echo "a100" ;;
    *"a40"*)                 echo "a40" ;;
    *"h100"*"nvl"*)          echo "h100-nvl" ;;
    *"h100"*"sxm"*)          echo "h100-sxm" ;;
    *"h100"*)                echo "h100" ;;
    *"h200"*)                echo "h200-sxm" ;;
    *"l40s"*)                echo "l40s" ;;
    *"l40"*)                 echo "l40" ;;
    *"l4"*)                  echo "l4" ;;
    *"v100"*"32gb"*)         echo "v100-32gb" ;;
    *"v100"*)                echo "v100" ;;
    *"a4000"*)               echo "rtxa4000" ;;
    *"a5000"*)               echo "rtxa5000" ;;
    *"a6000"*)               echo "rtxa6000" ;;
    *"rtx 2000 ada"*|*"rtx2000ada"*)  echo "rtx2000-ada" ;;
    *"rtx 4000 ada"*|*"rtx4000ada"*)  echo "rtx4000-ada" ;;
    *"rtx 6000 ada"*|*"rtx6000ada"*)  echo "rtx6000-ada" ;;
    *) echo "$raw" | tr -d ' -' | tr '[:upper:]' '[:lower:]' ;;
  esac
}

_label_gpu_nodes_for_skypilot() {
  local gpu_name=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
  fi
  if [[ -z "$gpu_name" ]]; then
    gpu_name=$(kubectl get nodes \
      -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.product}' 2>/dev/null \
      | tr '-' ' ' | xargs)
  fi
  if [[ -z "$gpu_name" ]]; then
    warn "Cannot detect GPU name — skipping SkyPilot node labels"
    return 0
  fi
  local skypilot_name
  skypilot_name=$(_map_nvidia_name_to_skypilot "$gpu_name")
  local gpu_count=1
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    [[ "${gpu_count}" -ge 1 ]] 2>/dev/null || gpu_count=1
  fi
  info "GPU node labeling: '${gpu_name}' → skypilot.co/accelerator=${skypilot_name} (count=${gpu_count})"
  local node
  for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    kubectl label node "${node}" \
      "skypilot.co/accelerator=${skypilot_name}" \
      "skypilot.co/accelerator-count=${gpu_count}" \
      --overwrite
  done
  ok "GPU nodes labeled for SkyPilot"
}

if kubectl get nodes >/dev/null 2>&1; then
  _label_gpu_nodes_for_skypilot
else
  warn "K3S API unreachable — skipping GPU node labeling"
fi

# ============================================================
# NAMESPACES + SECRETS
# ============================================================
sep
info "Creating namespaces"
kubectl create namespace kafka || true
kubectl create namespace ray || true
kubectl create namespace spark || true
kubectl create namespace monitoring || true
kubectl create namespace apps || true
kubectl create namespace airflow || true
kubectl create namespace skypilot || true


info "Creating secrets from .env"
kubectl create secret generic kafka-secret -n kafka --from-env-file=.env || true
kubectl create secret generic env-secret -n ray --from-env-file=.env || true
kubectl create secret generic env-secret -n spark --from-env-file=.env || true
kubectl create secret generic env-secret -n monitoring --from-env-file=.env || true
kubectl create secret generic env-secret -n apps --from-env-file=.env || true
kubectl create secret generic env-secret -n airflow --from-env-file=.env || true
kubectl create secret generic runpod-credentials \
  -n skypilot \
  --from-literal=api_key=$(grep RUNPOD_API_KEY .env | cut -d '=' -f2) || true

kubectl create secret generic vastai-credentials \
  -n skypilot \
  --from-literal=api_key=$(grep -E 'VASTAI_API_KEY|VAST_API_KEY' .env | head -n1 | cut -d '=' -f2) || true

kubectl create secret generic aws-credentials \
  -n skypilot \
  --from-literal=aws_access_key_id=$(grep AWS_ACCESS_KEY_ID .env | cut -d '=' -f2) \
  --from-literal=aws_secret_access_key=$(grep AWS_SECRET_ACCESS_KEY .env | cut -d '=' -f2) \
  --from-literal=aws_default_region=$(grep AWS_DEFAULT_REGION .env | cut -d '=' -f2 || echo us-east-1) || true

# Ingress basic-auth secret for Prometheus remote_write receiver.
# Requires METRICS_REMOTE_WRITE_USERNAME and METRICS_REMOTE_WRITE_PASSWORD in .env.
METRICS_REMOTE_WRITE_USERNAME="$(grep -m1 '^METRICS_REMOTE_WRITE_USERNAME=' .env | cut -d '=' -f2- || true)"
METRICS_REMOTE_WRITE_PASSWORD="$(grep -m1 '^METRICS_REMOTE_WRITE_PASSWORD=' .env | cut -d '=' -f2- || true)"
if [[ -n "${METRICS_REMOTE_WRITE_USERNAME}" && -n "${METRICS_REMOTE_WRITE_PASSWORD}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    RW_HTPASSWD="${METRICS_REMOTE_WRITE_USERNAME}:$(openssl passwd -apr1 "${METRICS_REMOTE_WRITE_PASSWORD}")"
    kubectl -n monitoring create secret generic prometheus-remote-write-auth \
      --from-literal=auth="${RW_HTPASSWD}" \
      --dry-run=client -o yaml | kubectl apply -f -
    ok "Secret prometheus-remote-write-auth upserted in monitoring namespace"
  else
    warn "openssl is not installed; skipping prometheus-remote-write-auth creation"
  fi
else
  warn "METRICS_REMOTE_WRITE_USERNAME/PASSWORD missing in .env; skipping prometheus-remote-write-auth"
fi
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
  helm upgrade --install my-strimzi-kafka-operator strimzi/strimzi-kafka-operator \
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

  # ------------------------------------------------------------
  # 9. Ensure a stable kafka-exporter Service exists (monitoring)
  # ------------------------------------------------------------
  # Prometheus scrapes my-kafka-cluster-kafka-exporter.kafka.svc.cluster.local:9404.
  # This Service provides a stable name/port even if the exporter pod labels differ.
  kubectl apply -f k3s/kafka/kafka-exporter-service.yaml

  ok "Kafka fully deployed 🚀"
}

# ============================================================
# RAY (PARALLEL)
# ============================================================
deploy_ray() {
  sep
  info "Deploying KubeRay Operator"
  helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --version 1.5.1 -n ray
  ok "KubeRay operator installed"
}

# ============================================================
# MLFLOW (PARALLEL)
# ============================================================
deploy_mlflow() {
  sep
  info "Deploying Postgres"
  helm upgrade --install postgres bitnami/postgresql \
    -n ray -f k3s/mlflow/postgres_values.yaml

  kubectl wait \
    --for=condition=Ready pod \
    -l app.kubernetes.io/instance=postgres \
    -n ray --timeout="${POSTGRES_TIMEOUT}"

  info "Deploying MLflow"
  helm upgrade --install my-mlflow community-charts/mlflow \
    -n ray -f k3s/mlflow/mlflow_values.yaml

  ok "MLflow deployed"
}

# ============================================================
# SPARK (PARALLEL)
# ============================================================
deploy_spark() {
  sep
  info "Deploying Spark Operator"
  helm upgrade --install my-spark-kubernetes-operator \
    spark-kubernetes-operator/spark-kubernetes-operator \
    --version 1.4.0 \
    -n spark -f k3s/spark/values.yaml
  ok "Spark operator deployed"
}

# ============================================================
# DSL APP (PARALLEL)
# ============================================================
deploy_dsl_app() {
  sep
  info "Deploying DSL App"

  kubectl apply -f k3s/app/dsl-app-rbac.yaml
  kubectl apply -f k3s/app/deployment.yaml
  kubectl apply -f k3s/app/services.yaml

  ok "DSL App deployed"
}

deploy_skypilot() {
  sep
  info "Deploying SkyPilot helm chart (k3s/sky/helm/skypilot)"
  #Este es un helm local
  helm upgrade --install my-skypilot ./k3s/sky/helm/skypilot \
    -n skypilot -f k3s/sky/helm/values.yaml
  #Este es el helm oficial
  #helm install my-skypilot skypilot/skypilot --version 0.12.0 -n skypilot -f k3s/sky/helm/values.yaml

  ok "SkyPilot deployed"
}

# ============================================================
# MONITORING (Prometheus + Grafana + kube-state-metrics)
# ============================================================
deploy_monitoring() {
  sep
  info "Deploying Monitoring Stack (Prometheus, Grafana, kube-state-metrics, node-exporter)"

  # Release any Retain-policy PVs left from a previous deploy cycle so new PVCs can bind.
  for pv in prometheus-pv-host; do
    phase="$(kubectl get pv "${pv}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "${phase}" == "Released" ]]; then
      info "Releasing stale PV ${pv} (was Retain/Released from previous cycle)"
      kubectl patch pv "${pv}" --type json -p '[{"op":"remove","path":"/spec/claimRef"}]'
    fi
  done

  # Apply all monitoring manifests
  info "Applying Prometheus"
  kubectl apply -f k3s/monitoring/prometheus/prometheus-stack.yaml

  info "Applying kube-state-metrics"
  kubectl apply -f k3s/monitoring/kube-state-metrics.yaml

  info "Applying node-exporter"
  kubectl apply -f k3s/monitoring/node-exporter.yaml

  info "Applying Grafana"
  kubectl apply -f k3s/monitoring/grafana/dashboards/
  kubectl apply -f k3s/monitoring/grafana/grafana.yaml


  # Always restart to ensure configmaps are loaded (dashboards / scrape config changes).
  info "Restarting Prometheus and Grafana to reload configuration"
  kubectl -n monitoring rollout restart deploy/prometheus || true
  kubectl -n monitoring rollout restart deploy/grafana || true

  # Wait for deployments
  info "Waiting for Prometheus"
  kubectl wait deployment/prometheus -n monitoring \
    --for=condition=Available --timeout=300s

  info "Waiting for Grafana"
  kubectl wait deployment/grafana -n monitoring \
    --for=condition=Available --timeout=300s

  info "Waiting for kube-state-metrics"
  kubectl wait deployment/kube-state-metrics -n monitoring \
    --for=condition=Available --timeout=300s

  ok "Monitoring stack deployed 📊"
  info "Grafana available at: http://grafana.localhost"
}

# ============================================================
# INGRESS (ABSOLUTELY LAST)
# ============================================================
# ============================================================
# AIRFLOW (PARALLEL)
# ============================================================
deploy_airflow() {
  sep
  info "Deploying Airflow"
  helm upgrade --install my-airflow apache-airflow/airflow -n airflow -f k3s/airflow/airflow_values.yaml

  info "Applying Airflow RBAC manifests"
  kubectl apply -f k3s/airflow/airflow-rbac.yaml
  kubectl apply -f k3s/airflow/rbac-node-reader.yaml

  # ── SkyPilot runner pod prerequisites ─────────────────────────────────────
  # sky-runner pods (KubernetesPodOperator) read ALL cloud credentials from
  # env-secret (already created above from .env). No extra secrets needed.
  #
  # Required keys in .env:
  #   AWS_ACCESS_KEY_ID        → boto3 reads natively (no file setup)
  #   AWS_SECRET_ACCESS_KEY    → boto3 reads natively
  #   AWS_DEFAULT_REGION       → e.g. us-east-1
  #   RUNPOD_API_KEY           → entrypoint writes ~/.runpod/config.toml
  #   VASTAI_API_KEY           → entrypoint writes ~/.config/vastai/vast_api_key
  #   MLFLOW_TRACKING_URI      → injected into sky Task envs
  #
  # Build the image (once, from REPO ROOT) before deploying:
  #   docker build -f k3s/sky/Dockerfile -t takenking9879/sky-runner:0.12.0 .
  #   docker push takenking9879/sky-runner:0.12.0
  #
  # Optional: if you need custom SkyPilot behavior config (~/.sky/config.yaml):
  #   kubectl create secret generic sky-config -n airflow \
  #     --from-file=config.yaml=~/.sky/config.yaml
  #   Then uncomment the sky-config volume/mount in sky-runner-pod.yaml and DAGs.

  ok "Airflow deployed"
}

deploy_cert_manager() {
  sep
  info "Deploying cert-manager"

  # Check that LETSENCRYPT_EMAIL is set — required for the ClusterIssuer.
  LETSENCRYPT_EMAIL="$(grep -m1 '^LETSENCRYPT_EMAIL=' .env | cut -d '=' -f2- || true)"
  if [[ -z "${LETSENCRYPT_EMAIL}" ]]; then
    warn "LETSENCRYPT_EMAIL is not set in .env — skipping cert-manager deploy"
    warn "Add LETSENCRYPT_EMAIL=you@example.com to .env and re-run with ENABLE_CERT_MANAGER=true"
    return 0
  fi

  helm repo add jetstack https://charts.jetstack.io --force-update
  helm upgrade --install cert-manager jetstack/cert-manager \
    --namespace cert-manager \
    --create-namespace \
    --version v1.17.2 \
    --set crds.enabled=true

  info "Waiting for cert-manager webhooks to be ready"
  kubectl wait deployment/cert-manager-webhook \
    -n cert-manager \
    --for=condition=Available \
    --timeout=120s

  info "Applying Let's Encrypt ClusterIssuer"
  # Substitute the email from .env before applying.
  LETSENCRYPT_EMAIL="${LETSENCRYPT_EMAIL}" \
    envsubst < k3s/cert-manager/clusterissuer.yaml | kubectl apply -f -

  ok "cert-manager deployed and ClusterIssuer ready"
}

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

  sleep 5 # small buffer time

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
info "Launching PARALLEL workloads (Ray / MLflow / Spark / App)"

PIDS=()

if [ "${ENABLE_MLFLOW}" = true ]; then
  deploy_mlflow & PIDS+=($!)
fi

if [ "${ENABLE_RAY}" = true ]; then
  deploy_ray & PIDS+=($!)
fi

if [ "${ENABLE_SPARK}" = true ]; then
  deploy_spark & PIDS+=($!)
fi

if [ "${ENABLE_APP}" = true ]; then
  deploy_dsl_app & PIDS+=($!)
fi

if [ "${ENABLE_SKYPILOT}" = true ]; then
  deploy_skypilot & PIDS+=($!)
fi

if [ "${ENABLE_AIRFLOW}" = true ]; then
  deploy_airflow & PIDS+=($!)
fi

info "Waiting for parallel jobs to finish"
wait "${PIDS[@]}"
ok "Parallel workloads completed"

if [ "${ENABLE_MONITORING}" = true ]; then
  deploy_monitoring
fi
if [ "${ENABLE_CERT_MANAGER}" = true ]; then
  deploy_cert_manager
fi
if [ "${ENABLE_INGRESS}" = true ]; then
  deploy_ingress
fi
sep
ok "DEPLOYMENT FINISHED SUCCESSFULLY"

sep
info "Next: start inference workloads with ./k3s/inference.sh"
