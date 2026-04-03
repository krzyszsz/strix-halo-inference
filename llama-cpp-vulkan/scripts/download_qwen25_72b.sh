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
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODEL_ROOT="$MODEL_ROOT" python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
import huggingface_hub.file_download as fd

fd.is_xet_available = lambda: False

repo_id = "bartowski/Qwen2.5-72B-Instruct-GGUF"
filename = "Qwen2.5-72B-Instruct-Q4_K_M.gguf"
local_dir = os.path.join(os.environ["MODEL_ROOT"], "qwen2.5-72b-instruct-gguf")

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Downloaded", path)
PY
