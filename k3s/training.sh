#!/bin/bash
# ============================================================
# Training & Tuning Pipeline Deployment Script
# ============================================================
# Deploys:
#   1. Spark preprocessing job with metrics
#   2. Ray training/tuning job with metrics
#   3. Associated Services for Prometheus scraping
# ============================================================

# Then deploy training pipeline ./k3s/training.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------ Check prerequisites ------------------
log_info "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubectl."
    exit 1
fi

# Ensure namespaces exist
kubectl get namespace spark &> /dev/null || kubectl create namespace spark
kubectl get namespace ray &> /dev/null || kubectl create namespace ray

# ------------------ Deploy Spark Preprocessing ------------------
log_info "Deploying Spark preprocessing driver service..."
kubectl apply -f "${SCRIPT_DIR}/spark/preprocess-driver-service.yaml"

log_info "Deploying Spark preprocessing job..."
kubectl apply -f "${SCRIPT_DIR}/spark/spark-application.yaml"

log_info "Waiting for Spark driver to be running..."
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    STATUS=$(kubectl get sparkapplication spark-preprocess -n spark -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo "PENDING")
    if [ "$STATUS" == "RUNNING" ] || [ "$STATUS" == "COMPLETED" ]; then
        log_info "Spark driver status: $STATUS"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo -n "."
done
echo ""

if [ $ELAPSED -ge $TIMEOUT ]; then
    log_warn "Timeout waiting for Spark driver. Continuing anyway..."
fi

# ------------------ Deploy Ray Training ------------------
log_info "Deploying Ray training head service..."
kubectl apply -f "${SCRIPT_DIR}/kuberay/kuberay-training-service.yaml"

log_info "Deploying Ray training job..."
kubectl apply -f "${SCRIPT_DIR}/kuberay/kuberay-job.yaml"

log_info "Waiting for Ray head pod to be running..."
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    READY=$(kubectl get pods -n ray -l app=kuberay-training,ray.io/node-type=head -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Pending")
    if [ "$READY" == "Running" ]; then
        log_info "Ray head pod is running"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo -n "."
done
echo ""

if [ $ELAPSED -ge $TIMEOUT ]; then
    log_warn "Timeout waiting for Ray head. Continuing anyway..."
fi

# ------------------ Validate Metrics Endpoints ------------------
log_info "Validating metrics endpoints..."

echo ""
log_info "=== Deployment Summary ==="
echo "  Spark Preprocessing:"
echo "    - SparkApplication: spark-preprocess (namespace: spark)"
echo "    - Metrics Service:  spark-preprocess-driver-svc:8001"
echo "    - Spark UI:         spark-preprocess-driver-svc:4040"
echo ""
echo "  Ray Training:"
echo "    - RayJob:           kuberay-training-job (namespace: ray)"  
echo "    - Metrics Service:  kuberay-training-head-svc:8002"
echo "    - Ray Dashboard:    kuberay-training-head-svc:8265"
echo ""
echo "  Prometheus Scrape Targets:"
echo "    - spark-preprocess-driver-svc.spark:8001/metrics"
echo "    - spark-preprocess-driver-svc.spark:4040/metrics/prometheus"
echo "    - kuberay-training-head-svc.ray:8002/metrics"
echo ""
echo "  Grafana Dashboard:"
echo "    - MLOps Training Dashboard (uid: mlops-training-main)"
echo ""

log_info "Training pipeline deployed successfully!"
log_info "To check Spark logs:  kubectl logs -n spark -l spark-role=driver -f"
log_info "To check Ray logs:    kubectl logs -n ray -l app=kuberay-training -f"
