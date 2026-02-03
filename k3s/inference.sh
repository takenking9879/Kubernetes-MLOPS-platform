#!/usr/bin/env bash
set -euo pipefail

# Run ./k3s/inference.sh AFTER ./k3s/deploy.sh
# Deploys the inference plane:
# - Ray Serve (RayService)
# - Spark Kafka inference (SparkApplication)
# - Spark driver Service (for Prometheus + Spark UI metrics)

NS_RAY="ray"
NS_SPARK="spark"

sep() { echo; echo "============================================================"; echo; }
info() { echo "👉 $*"; }
ok() { echo "✅ $*"; }

require_kubectl() {
  command -v kubectl >/dev/null 2>&1 || {
    echo "kubectl not found" >&2
    exit 1
  }
}

wait_for_http_200() {
  # Usage: wait_for_http_200 <pod> <namespace> <container> <url>
  local pod="$1" ns="$2" container="$3" url="$4"

  local deadline=$((SECONDS + 240))
  while [ $SECONDS -lt $deadline ]; do
    if kubectl -n "$ns" exec "$pod" -c "$container" -- bash -lc "python - <<'PY'
import requests
url='${url}'
payload={'data': [[0.0]*14]}
try:
  r = requests.post(url, json=payload, timeout=10)
  print(r.status_code)
  raise SystemExit(0 if r.status_code == 200 else 1)
except Exception:
  raise SystemExit(1)
PY" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

require_kubectl

sep
info "Ensuring namespaces exist"
kubectl create namespace "$NS_RAY" >/dev/null 2>&1 || true
kubectl create namespace "$NS_SPARK" >/dev/null 2>&1 || true
ok "Namespaces ready"

sep
info "Applying RayService (Ray Serve)"
kubectl apply -f k3s/kuberay/serving/rayservice-model-serving.yaml
ok "RayService applied"

sep
info "Applying Spark driver Service (metrics + Spark UI metrics servlet)"
kubectl apply -f k3s/spark/inference/driver-service.yaml
ok "Driver Service applied"

sep
info "Applying Spark Kafka inference SparkApplication"
kubectl apply -f k3s/spark/inference/spark-kafka-application.yaml
ok "SparkApplication applied"

sep
info "Waiting for Spark driver pod to be Ready"
# Matches the selector used by driver-service.yaml
kubectl -n "$NS_SPARK" wait --for=condition=Ready pod \
  -l spark-app-selector=spark-kafka-app-0 -l spark-role=driver \
  --timeout=300s
ok "Spark driver Ready"

sep
info "Waiting for Ray Serve /infer to return HTTP 200"
# Find head pod + exec into ray-head container.
RAY_HEAD_POD=$(kubectl -n "$NS_RAY" get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -z "$RAY_HEAD_POD" ]; then
  info "Could not find Ray head pod via label ray.io/node-type=head; skipping HTTP probe."
else
  if wait_for_http_200 "$RAY_HEAD_POD" "$NS_RAY" ray-head "http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer"; then
    ok "Ray Serve is healthy (HTTP 200)"
  else
    echo "Ray Serve did not become healthy within timeout" >&2
    exit 1
  fi
fi

sep
ok "INFERENCE DEPLOYMENT FINISHED"
info "Grafana: http://grafana.localhost"
info "Ray Serve: http://model-serving-serve-svc.ray.svc.cluster.local:8000/infer (in-cluster)"
