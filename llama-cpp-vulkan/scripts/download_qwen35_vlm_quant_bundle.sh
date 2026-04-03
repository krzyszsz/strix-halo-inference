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
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
else
  echo "HF token not found. Set HF_TOKEN or create $HF_TOKEN_FILE." >&2
  exit 1
fi

REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/research}"
mkdir -p "$REPORT_DIR"

print_download_plan() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local files=("$@")

  REPO_ID="$repo_id" FILES_CSV="$(IFS=,; echo "${files[*]}")" python3 - <<'PY'
import os
from huggingface_hub import HfApi

repo = os.environ["REPO_ID"]
files = [x for x in os.environ["FILES_CSV"].split(",") if x]
api = HfApi()
tree = {item.path: item for item in api.list_repo_tree(repo, repo_type="model", recursive=True, expand=True)}

total = 0
print(f"Plan for {repo}:")
for rel in files:
    node = tree.get(rel)
    size = getattr(node, "size", None) if node else None
    if size is None:
        size_text = "unknown"
    else:
        total += int(size)
        size_text = f"{size / (1024**3):.2f} GiB"
    print(f"  - {rel} ({size_text})")
print(f"  Total expected: {total / (1024**3):.2f} GiB")
PY
  echo "  destination: $local_dir"
}

download_repo_files() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local files=("$@")

  mkdir -p "$local_dir"
  echo "Downloading $repo_id -> $local_dir"

  if [ "${HF_DOWNLOAD_BIN[0]}" = "hf" ]; then
    "${HF_DOWNLOAD_BIN[@]}" "$repo_id" "${files[@]}" \
      --local-dir "$local_dir"
  else
    "${HF_DOWNLOAD_BIN[@]}" "$repo_id" "${files[@]}" \
      --local-dir "$local_dir" \
      --local-dir-use-symlinks False
  fi
}

assert_files_exist() {
  local local_dir="$1"
  shift
  local files=("$@")
  local missing=0

  for f in "${files[@]}"; do
    if [ ! -f "$local_dir/$f" ]; then
      echo "Missing downloaded file: $local_dir/$f" >&2
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    exit 1
  fi
}

verify_repo() {
  local repo_id="$1"
  local local_dir="$2"
  local out_json="$3"
  shift 3
  local includes=("$@")
  local manifest
  manifest="$(mktemp "${TMPDIR:-/tmp}/qwen35_verify_manifest.XXXXXX.json")"
  trap 'rm -f "$manifest"' RETURN

  REPO_ID="$repo_id" LOCAL_DIR="$local_dir" MANIFEST_PATH="$manifest" INCLUDES_CSV="$(IFS=,; echo "${includes[*]}")" \
    python3 - <<'PY'
import json
import os

repo_id = os.environ["REPO_ID"]
local_dir = os.environ["LOCAL_DIR"]
manifest_path = os.environ["MANIFEST_PATH"]
includes = [x for x in os.environ.get("INCLUDES_CSV", "").split(",") if x]

payload = [{
    "repo_id": repo_id,
    "local_dir": local_dir,
    "include": includes,
    "optional": False,
}]

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

  python3 "$REPO_ROOT/scripts/verify_hf_sha256.py" --manifest "$manifest" --strict > "$out_json"
  rm -f "$manifest"
  trap - RETURN
  echo "Checksum report: $out_json"
}

NINE_REPO="unsloth/Qwen3.5-9B-GGUF"
NINE_DIR="${NINE_DIR:-$MODEL_ROOT/qwen3.5-9b-gguf}"
NINE_PATTERNS=(
  "Qwen3.5-9B-Q8_0.gguf"
  "mmproj-F16.gguf"
)

TWENTY7_REPO="unsloth/Qwen3.5-27B-GGUF"
TWENTY7_DIR="${TWENTY7_DIR:-$MODEL_ROOT/qwen3.5-27b-gguf}"
TWENTY7_PATTERNS=(
  "Qwen3.5-27B-Q8_0.gguf"
  "mmproj-F16.gguf"
)

ONE22_REPO="unsloth/Qwen3.5-122B-A10B-GGUF"
ONE22_DIR="${ONE22_DIR:-$MODEL_ROOT/qwen3.5-122b-a10b-gguf}"
ONE22_PATTERNS=(
  "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
  "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf"
  "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf"
  "mmproj-F16.gguf"
)

print_download_plan "$NINE_REPO" "$NINE_DIR" "${NINE_PATTERNS[@]}"
download_repo_files "$NINE_REPO" "$NINE_DIR" "${NINE_PATTERNS[@]}"
assert_files_exist "$NINE_DIR" "${NINE_PATTERNS[@]}"
print_download_plan "$TWENTY7_REPO" "$TWENTY7_DIR" "${TWENTY7_PATTERNS[@]}"
download_repo_files "$TWENTY7_REPO" "$TWENTY7_DIR" "${TWENTY7_PATTERNS[@]}"
assert_files_exist "$TWENTY7_DIR" "${TWENTY7_PATTERNS[@]}"
print_download_plan "$ONE22_REPO" "$ONE22_DIR" "${ONE22_PATTERNS[@]}"
download_repo_files "$ONE22_REPO" "$ONE22_DIR" "${ONE22_PATTERNS[@]}"
assert_files_exist "$ONE22_DIR" "${ONE22_PATTERNS[@]}"

verify_repo "$NINE_REPO" "$NINE_DIR" "$REPORT_DIR/qwen35_9b_q8_verify.json" "${NINE_PATTERNS[@]}"
verify_repo "$TWENTY7_REPO" "$TWENTY7_DIR" "$REPORT_DIR/qwen35_27b_q8_verify.json" "${TWENTY7_PATTERNS[@]}"
verify_repo "$ONE22_REPO" "$ONE22_DIR" "$REPORT_DIR/qwen35_122b_a10b_q4km_verify.json" "${ONE22_PATTERNS[@]}"

echo "Done."
echo "Models downloaded:"
echo "  - $NINE_DIR"
echo "  - $TWENTY7_DIR"
echo "  - $ONE22_DIR"
echo "Verification reports:"
echo "  - $REPORT_DIR/qwen35_9b_q8_verify.json"
echo "  - $REPORT_DIR/qwen35_27b_q8_verify.json"
echo "  - $REPORT_DIR/qwen35_122b_a10b_q4km_verify.json"
