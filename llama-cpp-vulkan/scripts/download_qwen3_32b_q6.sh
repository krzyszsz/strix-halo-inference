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

huggingface-cli download gaianet/Qwen3-32B-GGUF \
  --include "Qwen3-32B-Q6_K.gguf" \
  --local-dir $MODEL_ROOT/qwen3-32b-gguf \
  --local-dir-use-symlinks False
