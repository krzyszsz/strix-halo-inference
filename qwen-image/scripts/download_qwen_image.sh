#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen-Image}"
TARGET_DIR="${TARGET_DIR:-$MODEL_ROOT/qwen-image}"
HF_KEY_PATH="${HF_KEY_PATH:-$HF_TOKEN_FILE}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -f "$HF_KEY_PATH" ]; then
  HF_TOKEN="$(cat "$HF_KEY_PATH")"
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if ! mkdir -p "$TARGET_DIR" 2>/dev/null; then
  sudo mkdir -p "$TARGET_DIR"
fi
if [ ! -w "$TARGET_DIR" ]; then
  sudo chown -R "$(id -u):$(id -g)" "$TARGET_DIR"
fi

huggingface-cli download "$MODEL_REPO" \
  --local-dir "$TARGET_DIR" \
  --local-dir-use-symlinks False
