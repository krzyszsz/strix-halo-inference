#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"
ROUND_ROOT="${ROUND_ROOT:-$REPO_ROOT/reports/retest_2026-02-10_ctx}"
mkdir -p "$ROUND_ROOT"

run_one() {
  local alias="$1"
  local model_path="$2"
  local prompt_mode="$3"
  local port="$4"
  local min_ctx="$5"
  local cap_ctx="$6"
  local max_tokens="$7"

  local log_path="$ROUND_ROOT/${alias}_search.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${alias} search (log=${log_path})"

  env \
    MODEL_ALIAS="$alias" \
    MODEL_PATH="$model_path" \
    PROMPT_MODE="$prompt_mode" \
    PORT="$port" \
    MIN_CTX="$min_ctx" \
    MAX_CTX_CAP="$cap_ctx" \
    MAX_TOKENS="$max_tokens" \
    PRECISION=1024 \
    "$SCRIPT_DIR/search_max_ctx_binary.sh" \
    > "$log_path" 2>&1

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] completed ${alias} search"
}

run_one \
  "qwen3_next_80b_q5" \
  "$MODEL_ROOT/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf" \
  "text" \
  "8103" \
  "32768" \
  "196608" \
  "1024"

run_one \
  "gpt_oss_120b_mxfp4" \
  "$MODEL_ROOT/gpt-oss-120b-gguf/gpt-oss-120b-mxfp4-00001-of-00003.gguf" \
  "text" \
  "8105" \
  "4096" \
  "131072" \
  "1024"

run_one \
  "qwen3_coder_next_q5" \
  "$MODEL_ROOT/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf" \
  "coding" \
  "8104" \
  "8192" \
  "196608" \
  "1536"

run_one \
  "qwen25_coder_32b_q4" \
  "$MODEL_ROOT/qwen2.5-coder-32b-instruct-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf" \
  "coding" \
  "8106" \
  "8192" \
  "131072" \
  "1536"

echo "Context capacity round completed. Summary: $ROUND_ROOT/summary.tsv"
