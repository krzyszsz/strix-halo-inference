#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  docker() { sudo docker "$@"; }
else
  docker() { command docker "$@"; }
fi

docker build -t voxtral-rocm:latest -f "$REPO_ROOT/audio/Dockerfile.voxtral.rocm" "$REPO_ROOT/audio"
