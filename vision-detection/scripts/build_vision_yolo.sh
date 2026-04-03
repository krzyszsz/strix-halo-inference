#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi

$SUDO docker build -t vision-yolo-rocm:latest -f "$REPO_ROOT/vision-detection/Dockerfile" "$REPO_ROOT/vision-detection"
