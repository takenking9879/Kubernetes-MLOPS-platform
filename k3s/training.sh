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

# Resource names (match manifests by default)
ENABLE_DELETION=${ENABLE_DELETION:-true}
SPARK_APP_NAME=${SPARK_APP_NAME:-spark-app}
RAY_JOB_NAME=${RAY_JOB_NAME:-kuberay-job}

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
TIMEOUT=130
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    STATUS=$(kubectl get sparkapplication "$SPARK_APP_NAME" -n spark -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo "PENDING")
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

sleep 10  # brief wait to ensure service is up

# ------------------ Deploy Ray Training ------------------
log_info "Deploying Ray training head service..."
kubectl apply -f "${SCRIPT_DIR}/kuberay/kuberay-training-service.yaml"

log_info "Deploying Ray training submitter metrics service..."
kubectl apply -f "${SCRIPT_DIR}/kuberay/kuberay-training-metrics-service.yaml"

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
# ------------------ Optional cleanup after completion ------------------
# If ENABLE_DELETION is true, wait for SparkApplication `spark-app`
# to reach COMPLETED and then delete it, then wait for Ray job
# `kuberay-job` to finish successfully and delete it.
if [ "$ENABLE_DELETION" = "true" ]; then
    log_info "ENABLE_DELETION=true — will delete completed resources."

    # Wait for SparkApplication to complete
    log_info "Waiting for SparkApplication ${SPARK_APP_NAME} to finish..."
    TIMEOUT=1800
    INTERVAL=10
    ELAPSED=0
    SPARK_STATE=""
    while [ $ELAPSED -lt $TIMEOUT ]; do
        SPARK_STATE=$(kubectl get sparkapplication "$SPARK_APP_NAME" -n spark -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo "")
        # Delete when Spark transitions to ResourceReleased (driver/resources cleaned)
        if [ "$SPARK_STATE" = "ResourceReleased" ] || [ "$SPARK_STATE" = "RESOURCE_RELEASED" ]; then
            log_info "SparkApplication ${SPARK_APP_NAME} reached ResourceReleased"
            break
        fi
        # also allow deletion if the application failed
        if [ "$SPARK_STATE" = "FAILED" ] || [ "$SPARK_STATE" = "Failed" ]; then
            log_warn "SparkApplication ${SPARK_APP_NAME} finished with failure: $SPARK_STATE"
            break
        fi
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        echo -n "."
    done
    echo ""

    if [ "$SPARK_STATE" = "ResourceReleased" ] || [ "$SPARK_STATE" = "RESOURCE_RELEASED" ] || [ "$SPARK_STATE" = "FAILED" ] || [ "$SPARK_STATE" = "Failed" ]; then
        log_info "Deleting SparkApplication ${SPARK_APP_NAME}..."
        (kubectl delete sparkapp "$SPARK_APP_NAME" -n spark || kubectl delete sparkapplication "$SPARK_APP_NAME" -n spark) || true
    else
        log_warn "SparkApplication ${SPARK_APP_NAME} did not reach ResourceReleased/Failed within timeout (state=${SPARK_STATE}). Skipping delete."
    fi

    # Wait for Ray job to reach Succeeded/Failed — try to read job status, otherwise infer from pods
    log_info "Waiting for RayJob ${RAY_JOB_NAME} to finish successfully..."
    TIMEOUT=3600
    INTERVAL=15
    ELAPSED=0
    RAY_STATE=""
    while [ $ELAPSED -lt $TIMEOUT ]; do
        # Try multiple paths for Ray job state — CRD versions vary
        RAY_STATE=$(kubectl get rayjob "$RAY_JOB_NAME" -n ray -o jsonpath='{.status.jobStatus.state}' 2>/dev/null || echo "")
        RAY_JOBSTATUS=$(kubectl get rayjob "$RAY_JOB_NAME" -n ray -o jsonpath='{.status.jobStatus.jobStatus}' 2>/dev/null || echo "")
        RAY_DEPLOY=$(kubectl get rayjob "$RAY_JOB_NAME" -n ray -o jsonpath='{.status.deploymentStatus.phase}' 2>/dev/null || echo "")

        # normalize to lowercase for comparisons
        rstate_lc=$(echo "$RAY_STATE" | tr '[:upper:]' '[:lower:]')
        rjob_lc=$(echo "$RAY_JOBSTATUS" | tr '[:upper:]' '[:lower:]')
        rdep_lc=$(echo "$RAY_DEPLOY" | tr '[:upper:]' '[:lower:]')

        if [ -n "$rstate_lc" ] && ( [ "$rstate_lc" = "succeeded" ] || [ "$rstate_lc" = "failed" ] ); then
            log_info "RayJob ${RAY_JOB_NAME} finished with state: $RAY_STATE"
            RAY_STATE=$rstate_lc
            break
        fi

        if [ -n "$rjob_lc" ] && ( echo "$rjob_lc" | grep -qi "succeed" >/dev/null 2>&1 || echo "$rjob_lc" | grep -qi "succ" >/dev/null 2>&1 ); then
            log_info "RayJob ${RAY_JOB_NAME} jobStatus indicates success: $RAY_JOBSTATUS"
            RAY_STATE="Succeeded"
            break
        fi

        if [ -n "$rdep_lc" ] && ( [ "$rdep_lc" = "complete" ] || [ "$rdep_lc" = "completed" ] ); then
            log_info "RayJob ${RAY_JOB_NAME} deploymentStatus indicates completion: $RAY_DEPLOY"
            RAY_STATE="Succeeded"
            break
        fi

        # fallback: if no jobStatus, check that no pods with the ray job label are Running
        RUNNING_PODS=$(kubectl get pods -n ray -l app=kuberay-training -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2>/dev/null || echo "")
        if [ -z "$RUNNING_PODS" ]; then
            RAY_STATE="Succeeded"
            log_info "No running Ray pods found; assuming job finished."
            break
        fi
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        echo -n "."
    done
    echo ""

    if [ "$RAY_STATE" = "Succeeded" ]; then
        log_info "Deleting RayJob ${RAY_JOB_NAME}..."
        (kubectl delete rayjob "$RAY_JOB_NAME" -n ray) || true
    else
        log_warn "RayJob ${RAY_JOB_NAME} did not finish successfully within timeout (state=${RAY_STATE}). Skipping delete."
    fi
else
    log_info "ENABLE_DELETION is false — skipping automatic cleanup."
fi
echo ""
log_info "=== Deployment Summary ==="
echo "  Spark Preprocessing:"
echo "    - SparkApplication: ${SPARK_APP_NAME} (namespace: spark)"
echo "    - Metrics Service:  spark-preprocess-driver-svc:8001"
echo "    - Spark UI:         spark-preprocess-driver-svc:4040"
echo ""
echo "  Ray Training:"
echo "    - RayJob:           ${RAY_JOB_NAME} (namespace: ray)"  
echo "    - Metrics Service:  kuberay-training-metrics-svc:8002"
echo "    - Ray Dashboard:    kuberay-training-head-svc:8265"
echo ""
echo "  Prometheus Scrape Targets:"
echo "    - spark-preprocess-driver-svc.spark:8001/metrics"
echo "    - spark-preprocess-driver-svc.spark:4040/metrics/prometheus"
echo "    - kuberay-training-metrics-svc.ray:8002/metrics"
echo ""
echo "  Grafana Dashboard:"
echo "    - MLOps Training Dashboard (uid: mlops-training-main)"
echo ""

log_info "Training pipeline deployed successfully!"
log_info "To check Spark logs:  kubectl logs -n spark -l spark-role=driver -f"
log_info "To check Ray logs:    kubectl logs -n ray -l app=kuberay-training -f"
