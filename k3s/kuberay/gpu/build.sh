#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="takenking9879"

PUSH=true
TARGET="all"

# Parse args
for arg in "$@"; do
  case "$arg" in
    train|llm|all)
      TARGET="$arg"
      ;;
    --no-push)
      PUSH=false
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [train|llm|all] [--no-push]"
      exit 1
      ;;
  esac
done

build() {
  local name=$1 tag=$2 dockerfile=$3
  echo ""
  echo "==> Building ${REGISTRY}/${name}:${tag}"
  docker build -t "${REGISTRY}/${name}:${tag}" -f "${DIR}/${dockerfile}" "${DIR}"
}

push() {
  local name=$1 tag=$2
  echo "==> Pushing ${REGISTRY}/${name}:${tag}"
  docker push "${REGISTRY}/${name}:${tag}"
}

run_target() {
  local name=$1 tag=$2 dockerfile=$3
  build "$name" "$tag" "$dockerfile"
  if [[ "$PUSH" == "true" ]]; then
    push "$name" "$tag"
  fi
}

case "$TARGET" in
  train)
    run_target "ray-train" "2.53.0" "Dockerfile"
    ;;
  llm)
    run_target "ray-llm" "2.53.0" "Dockerfile.llm"
    ;;
  all)
    run_target "ray-train" "2.53.0" "Dockerfile"
    run_target "ray-llm" "2.53.0" "Dockerfile.llm"
    ;;
esac

echo ""
echo "Done. Local images:"
docker images | grep -E "(ray-train|ray-llm)" || true