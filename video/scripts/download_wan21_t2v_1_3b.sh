#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

HF_KEY_PATH="${HF_KEY_PATH:-$HF_TOKEN_FILE}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -f "$HF_KEY_PATH" ]; then
  HF_TOKEN="$(cat "$HF_KEY_PATH")"
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
LOCAL_DIR="${LOCAL_DIR:-$MODEL_ROOT/wan21-t2v-1.3b-diffusers}"

if [ -w "$(dirname "$LOCAL_DIR")" ] || [ -d "$LOCAL_DIR" ] && [ -w "$LOCAL_DIR" ]; then
  huggingface-cli download "$MODEL_REPO" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False
else
  sudo mkdir -p "$LOCAL_DIR"
  sudo HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    huggingface-cli download "$MODEL_REPO" \
      --local-dir "$LOCAL_DIR" \
      --local-dir-use-symlinks False
fi
