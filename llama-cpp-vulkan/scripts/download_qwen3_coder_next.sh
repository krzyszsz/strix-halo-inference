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

huggingface-cli download Qwen/Qwen3-Coder-Next-GGUF \
  --include "Qwen3-Coder-Next-Q5_K_M/*.gguf" \
  --local-dir $MODEL_ROOT/qwen3-coder-next-gguf \
  --local-dir-use-symlinks False
