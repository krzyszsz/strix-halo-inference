#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

docker_cmd() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

IMAGE_TAG="${IMAGE_TAG:-mcp-image-tools:1.0}"
DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/mcp/docker/image-tools/Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$REPO_ROOT/mcp/docker/image-tools}"

if [ ! -f "$DOCKERFILE" ]; then
  echo "Missing Dockerfile: $DOCKERFILE" >&2
  exit 2
fi

docker_cmd build \
  -f "$DOCKERFILE" \
  -t "$IMAGE_TAG" \
  "$BUILD_CONTEXT"

echo "Built image: $IMAGE_TAG"

