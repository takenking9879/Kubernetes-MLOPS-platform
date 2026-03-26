#!/usr/bin/env bash
# build.sh — Build and push Ray Docker images to DockerHub
#
# Images:
#   ray-train:2.53.0  — Tabular training (ANN/SSM/BAE/XGBoost), used by KubeRay + AWS multi-node
#   ray-llm:2.53.0    — LLM fine-tuning + vLLM serving, used by AWS single/multi-node
#
# Provider-native images (RunPod / Vast.ai) are NOT built here.
# They use provider base templates directly in SkyPilot YAMLs:
#   RunPod: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   Vast.ai: vastai/pytorch:2.10.0-cu130-cuda-13.1-mini-py311-2026-03-19
#
# Usage:
#   ./build.sh            # build + push both images
#   ./build.sh train      # build + push ray-train only
#   ./build.sh llm        # build + push ray-llm only
#   ./build.sh --no-push  # build only, skip push

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="takenking9879"
PUSH=true
TARGET="${1:-all}"

if [[ "${TARGET}" == "--no-push" ]]; then
  PUSH=false
  TARGET="all"
fi

build_and_push() {
  local name=$1 tag=$2 dockerfile=$3
  echo ""
  echo "==> Building ${REGISTRY}/${name}:${tag} from ${dockerfile}"
  docker build -t "${REGISTRY}/${name}:${tag}" -f "${DIR}/${dockerfile}" "${DIR}"
  if [[ "${PUSH}" == "true" ]]; then
    echo "==> Pushing ${REGISTRY}/${name}:${tag}"
    docker push "${REGISTRY}/${name}:${tag}"
  fi
}

case "${TARGET}" in
  train|all)
    build_and_push "ray-train" "2.53.0" "Dockerfile"
    ;;&
  llm|all)
    build_and_push "ray-llm" "2.53.0" "Dockerfile.llm"
    ;;
  *)
    echo "Usage: $0 [train|llm|all|--no-push]" >&2
    exit 1
    ;;
esac

echo ""
echo "Done. Images:"
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" \
  | grep -E "(ray-train|ray-llm)" || true
