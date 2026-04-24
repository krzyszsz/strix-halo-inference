#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

require_cmd python3

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_BIN=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_BIN=(huggingface-cli download)
else
  echo "Missing required command: hf (preferred) or huggingface-cli" >&2
  exit 1
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
elif [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
else
  echo "HF token not found. Set HF_TOKEN or create $HF_TOKEN_FILE." >&2
  exit 1
fi

REPO_ID="${REPO_ID:-batiai/Qwen3.6-27B-GGUF}"
LOCAL_DIR="${LOCAL_DIR:-$MODEL_ROOT/qwen3.6-27b-gguf}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/research}"
MODEL_FILE="${MODEL_FILE:-Qwen-Qwen3.6-27B-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen-Qwen3.6-27B-BF16.gguf}"
VERIFY_JSON="${VERIFY_JSON:-$REPORT_DIR/qwen36_27b_q4km_verify.json}"

mkdir -p "$LOCAL_DIR" "$REPORT_DIR"
export REPO_ID LOCAL_DIR MODEL_FILE MMPROJ_FILE VERIFY_JSON

python3 - <<'PY'
from huggingface_hub import HfApi
import os

repo = os.environ["REPO_ID"]
files = [os.environ["MODEL_FILE"], os.environ["MMPROJ_FILE"]]
api = HfApi()
tree = {item.path: item for item in api.list_repo_tree(repo, repo_type="model", recursive=True, expand=True)}
total = 0
print(f"Download plan for {repo}:")
for rel in files:
    node = tree.get(rel)
    if node is None:
        raise SystemExit(f"Missing expected file in repo: {rel}")
    size = getattr(node, "size", 0) or 0
    total += size
    print(f"  - {rel}: {size / (1024 ** 3):.2f} GiB")
print(f"  total: {total / (1024 ** 3):.2f} GiB")
PY

echo "Downloading $REPO_ID -> $LOCAL_DIR"
if [ "${HF_DOWNLOAD_BIN[0]}" = "hf" ]; then
  "${HF_DOWNLOAD_BIN[@]}" "$REPO_ID" "$MODEL_FILE" "$MMPROJ_FILE" --local-dir "$LOCAL_DIR"
else
  "${HF_DOWNLOAD_BIN[@]}" "$REPO_ID" "$MODEL_FILE" "$MMPROJ_FILE" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False
fi

for f in "$MODEL_FILE" "$MMPROJ_FILE"; do
  if [ ! -f "$LOCAL_DIR/$f" ]; then
    echo "Missing downloaded file: $LOCAL_DIR/$f" >&2
    exit 1
  fi
done

manifest="$(mktemp "${TMPDIR:-/tmp}/qwen36_verify_manifest.XXXXXX.json")"
trap 'rm -f "$manifest"' EXIT

REPO_ID="$REPO_ID" LOCAL_DIR="$LOCAL_DIR" MODEL_FILE="$MODEL_FILE" MMPROJ_FILE="$MMPROJ_FILE" MANIFEST="$manifest" python3 - <<'PY'
import json
import os

payload = [{
    "repo_id": os.environ["REPO_ID"],
    "local_dir": os.environ["LOCAL_DIR"],
    "include": [os.environ["MODEL_FILE"], os.environ["MMPROJ_FILE"]],
    "optional": False,
}]
with open(os.environ["MANIFEST"], "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

python3 "$REPO_ROOT/scripts/verify_hf_sha256.py" --manifest "$manifest" --strict > "$VERIFY_JSON"

echo "Done."
echo "Model: $LOCAL_DIR/$MODEL_FILE"
echo "mmproj: $LOCAL_DIR/$MMPROJ_FILE"
echo "Verification report: $VERIFY_JSON"
