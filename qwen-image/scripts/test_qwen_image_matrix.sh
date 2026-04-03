#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/qwen-image/out/matrix_${RUN_ID}}"
PROMPT="${PROMPT:-A vivid photorealistic still life of translucent glass fruit on a reflective table, soft studio light}"
MODEL_ID="${MODEL_ID:-$MODEL_ROOT/qwen-image-2512}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-qwen-image-rocm:latest}"

mkdir -p "$OUT_ROOT"
REPORT_TSV="$OUT_ROOT/report.tsv"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

cat >"$REPORT_TSV" <<'EOF'
case	width	height	steps	status	elapsed_seconds	output_path
EOF

run_case() {
  local case_name="$1"
  local width="$2"
  local height="$3"
  local steps="$4"
  local out_path="$OUT_ROOT/${case_name}.png"
  local log_path="$LOG_DIR/${case_name}.log"
  local start_ts end_ts elapsed

  start_ts="$(date +%s)"
  if env \
      MODEL_ID="$MODEL_ID" \
      WIDTH="$width" \
      HEIGHT="$height" \
      STEPS="$steps" \
      PROMPT="$PROMPT" \
      OUT_PATH="$out_path" \
      CONTAINER_IMAGE="$CONTAINER_IMAGE" \
      bash "$SCRIPT_DIR/test_qwen_image.sh" >"$log_path" 2>&1; then
    end_ts="$(date +%s)"
    elapsed="$((end_ts - start_ts))"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$case_name" "$width" "$height" "$steps" "ok" "$elapsed" "$out_path" >>"$REPORT_TSV"
  else
    end_ts="$(date +%s)"
    elapsed="$((end_ts - start_ts))"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$case_name" "$width" "$height" "$steps" "fail" "$elapsed" "$out_path" >>"$REPORT_TSV"
    return 1
  fi
}

run_case tiny_256_s4 256 256 4
run_case small_384_s8 384 384 8
run_case medium_512_s12 512 512 12

echo "Saved report: $REPORT_TSV"
