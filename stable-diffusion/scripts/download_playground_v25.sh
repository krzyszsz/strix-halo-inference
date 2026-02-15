#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

# Not gated, but allow using the same external token mechanism as other downloads.
HF_KEY_PATH="${HF_KEY_PATH:-$HF_TOKEN_FILE}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -f "$HF_KEY_PATH" ]; then
  HF_TOKEN="$(cat "$HF_KEY_PATH")"
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# This repo contains redundant weight formats and single-file checkpoints.
# For diffusers usage in this repo, we only need the standard `.safetensors` component weights + configs.
hf download playgroundai/playground-v2.5-1024px-aesthetic \
  --local-dir "$MODEL_ROOT/playground-v2.5-1024px-aesthetic" \
  --exclude "playground-v2.5-1024px-aesthetic*.safetensors" "*.bin" "*fp16.safetensors"
