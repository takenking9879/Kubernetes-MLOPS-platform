#!/bin/bash
# Usage: ./k3s/exec-into-skypilot.sh

POD_PATTERN="my-skypilot-api-server-"
NAMESPACE="skypilot"

POD_NAME=$(kubectl get pods -n "$NAMESPACE" -o name | grep "$POD_PATTERN" | head -n1)

if [ -z "$POD_NAME" ]; then
  echo "No pod found matching pattern: $POD_PATTERN"
  exit 1
fi

POD_NAME="${POD_NAME#pods/}"

echo "Entering pod: $POD_NAME"
kubectl exec -it -n "$NAMESPACE" "$POD_NAME" -- /bin/bash