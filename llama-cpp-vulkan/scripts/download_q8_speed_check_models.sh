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

require_cmd aria2c
require_cmd python3

QWEN_DIR="${QWEN_DIR:-$MODEL_ROOT/qwen3.6-27b-q8-gguf}"
GEMMA_DIR="${GEMMA_DIR:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/research}"
DATE_TAG="${DATE_TAG:-$(date -u +%F)}"

mkdir -p "$QWEN_DIR" "$GEMMA_DIR" "$REPORT_DIR"

TOKEN_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_HEADER=(--header="Authorization: Bearer $HF_TOKEN")
elif [ -f "$HF_TOKEN_FILE" ]; then
  TOKEN_HEADER=(--header="Authorization: Bearer $(tr -d '\r\n' < "$HF_TOKEN_FILE")")
fi

ARIA2_COMMON=(
  --console-log-level=notice
  --summary-interval=30
  --max-connection-per-server="${ARIA2_SPLIT:-8}"
  --split="${ARIA2_SPLIT:-8}"
  --min-split-size="${ARIA2_MIN_SPLIT_SIZE:-64M}"
  --file-allocation=none
  --continue=true
  --timeout=30
  --connect-timeout=30
  --retry-wait=5
  --max-tries=0
)

download_file() {
  local repo="$1"
  local rel="$2"
  local sha256="$3"
  local out_dir="$4"
  local out_name="$5"

  echo "Downloading/verifying $repo/$rel -> $out_dir/$out_name"
  aria2c "${ARIA2_COMMON[@]}" "${TOKEN_HEADER[@]}" \
    --checksum="sha-256=$sha256" \
    -d "$out_dir" \
    -o "$out_name" \
    "https://huggingface.co/${repo}/resolve/main/${rel}?download=true"
}

download_file \
  "eaddario/Qwen3.6-27B-GGUF" \
  "Qwen3.6-27B-Q8_0.gguf" \
  "e74f6ddc6f1ea2d811cadf70128eb52de514abcf9c076e9fbb0dfe80e408bfff" \
  "$QWEN_DIR" \
  "Qwen3.6-27B-Q8_0.gguf"

download_file \
  "eaddario/Qwen3.6-27B-GGUF" \
  "mmproj-Qwen3.6-27B-F16.gguf" \
  "26b00d800d2853627d09f0caabde10e79a1a6e5e5dc589fa62deeb1b14f3c673" \
  "$QWEN_DIR" \
  "mmproj-Qwen3.6-27B-F16.gguf"

download_file \
  "ggml-org/gemma-4-26b-a4b-it-GGUF" \
  "gemma-4-26B-A4B-it-Q8_0.gguf" \
  "69e2d9d1381ff60e862c18faf4ddaadc2ca9f945710cc6f81c40f0e9f07827c3" \
  "$GEMMA_DIR" \
  "gemma-4-26B-A4B-it-Q8_0.gguf"

python3 "$REPO_ROOT/scripts/verify_hf_sha256.py" --strict \
  --repo eaddario/Qwen3.6-27B-GGUF \
  --local-dir "$QWEN_DIR" \
  --include Qwen3.6-27B-Q8_0.gguf \
  --include mmproj-Qwen3.6-27B-F16.gguf \
  > "$REPORT_DIR/qwen36_27b_q8_verify_${DATE_TAG}.json"

python3 "$REPO_ROOT/scripts/verify_hf_sha256.py" --strict \
  --repo ggml-org/gemma-4-26b-a4b-it-GGUF \
  --local-dir "$GEMMA_DIR" \
  --include gemma-4-26B-A4B-it-Q8_0.gguf \
  > "$REPORT_DIR/gemma4_q8_verify_${DATE_TAG}.json"

echo "Done."
echo "Qwen verification: $REPORT_DIR/qwen36_27b_q8_verify_${DATE_TAG}.json"
echo "Gemma verification: $REPORT_DIR/gemma4_q8_verify_${DATE_TAG}.json"
