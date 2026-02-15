#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

MODEL_REPO="${MODEL_REPO:-black-forest-labs/FLUX.2-klein-9B}"
TARGET_DIR="${TARGET_DIR:-$MODEL_ROOT/flux2-klein-9b}"
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
if [ ! -w "$(dirname "$TARGET_DIR")" ]; then
  sudo chown "$(id -u):$(id -g)" "$(dirname "$TARGET_DIR")"
fi

if command -v hf >/dev/null 2>&1; then
  if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
    hf auth whoami >/dev/null 2>&1 || {
      echo "ERROR: no HF auth found. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or provide key at HF_KEY_PATH=$HF_KEY_PATH" >&2
      exit 2
    }
  fi
  # Do not pass tokens on the command line; rely on HF_TOKEN env var / cached auth.
  hf download "$MODEL_REPO" --repo-type model --local-dir "$TARGET_DIR"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_REPO" \
    --local-dir "$TARGET_DIR" \
    --local-dir-use-symlinks False
else
  echo "ERROR: missing Hugging Face CLI ('hf' or 'huggingface-cli')." >&2
  exit 2
fi
