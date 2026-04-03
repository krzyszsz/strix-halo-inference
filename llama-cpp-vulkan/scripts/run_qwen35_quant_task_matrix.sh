#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

run_suite() {
  local model_path="$1"
  local mmproj_path="$2"
  local port="$3"
  local model_tag="$4"
  local ctx_size="$5"
  local gpu_layers="$6"
  local mem_limit="$7"
  local mem_reservation="$8"
  local max_tokens="$9"

  if [ ! -f "$model_path" ] || [ ! -f "$mmproj_path" ]; then
    echo "Skipping $model_tag (missing files)." >&2
    [ -f "$model_path" ] || echo "  missing model: $model_path" >&2
    [ -f "$mmproj_path" ] || echo "  missing mmproj: $mmproj_path" >&2
    return 0
  fi

  RUN_WITH_WATCHDOG=1 \
  RUN_MAX_SECONDS=3600 \
  RUN_IDLE_SECONDS=900 \
  RUN_WATCH_PATH="$REPO_ROOT/llama-cpp-vulkan/out/qwen35-task-suite" \
    "$REPO_ROOT/scripts/run_memsafe.sh" \
    env MODEL_PATH="$model_path" MMPROJ_PATH="$mmproj_path" PORT="$port" MODEL_TAG="$model_tag" \
        CTX_SIZE="$ctx_size" GPU_LAYERS="$gpu_layers" MAX_TOKENS="$max_tokens" \
        MEM_LIMIT="$mem_limit" MEMORY_SWAP="$mem_limit" MEM_RESERVATION="$mem_reservation" \
      bash "$REPO_ROOT/llama-cpp-vulkan/scripts/run_qwen35_vlm_task_suite.sh"
}

CTX_9B="${CTX_9B:-32768}"
GPU_LAYERS_9B="${GPU_LAYERS_9B:-999}"
MEM_LIMIT_9B="${MEM_LIMIT_9B:-75g}"
MEM_RESERVATION_9B="${MEM_RESERVATION_9B:-67g}"
MAX_TOKENS_9B="${MAX_TOKENS_9B:-768}"

CTX_27B="${CTX_27B:-16384}"
GPU_LAYERS_27B="${GPU_LAYERS_27B:-999}"
MEM_LIMIT_27B="${MEM_LIMIT_27B:-75g}"
MEM_RESERVATION_27B="${MEM_RESERVATION_27B:-67g}"
MAX_TOKENS_27B="${MAX_TOKENS_27B:-768}"

CTX_122B="${CTX_122B:-8192}"
GPU_LAYERS_122B="${GPU_LAYERS_122B:-0}"
MEM_LIMIT_122B="${MEM_LIMIT_122B:-85g}"
MEM_RESERVATION_122B="${MEM_RESERVATION_122B:-76g}"
MAX_TOKENS_122B="${MAX_TOKENS_122B:-512}"

run_suite \
  "$MODEL_ROOT/qwen3.5-9b-gguf/Qwen3.5-9B-Q8_0.gguf" \
  "$MODEL_ROOT/qwen3.5-9b-gguf/mmproj-F16.gguf" \
  8131 \
  "qwen35_9b_q8" \
  "$CTX_9B" \
  "$GPU_LAYERS_9B" \
  "$MEM_LIMIT_9B" \
  "$MEM_RESERVATION_9B" \
  "$MAX_TOKENS_9B"

run_suite \
  "$MODEL_ROOT/qwen3.5-27b-gguf/Qwen3.5-27B-Q8_0.gguf" \
  "$MODEL_ROOT/qwen3.5-27b-gguf/mmproj-F16.gguf" \
  8132 \
  "qwen35_27b_q8" \
  "$CTX_27B" \
  "$GPU_LAYERS_27B" \
  "$MEM_LIMIT_27B" \
  "$MEM_RESERVATION_27B" \
  "$MAX_TOKENS_27B"

run_suite \
  "$MODEL_ROOT/qwen3.5-122b-a10b-gguf/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf" \
  "$MODEL_ROOT/qwen3.5-122b-a10b-gguf/mmproj-F16.gguf" \
  8133 \
  "qwen35_122b_a10b_q4km" \
  "$CTX_122B" \
  "$GPU_LAYERS_122B" \
  "$MEM_LIMIT_122B" \
  "$MEM_RESERVATION_122B" \
  "$MAX_TOKENS_122B"

echo "Done. Outputs (if runs succeeded): $REPO_ROOT/llama-cpp-vulkan/out/qwen35-task-suite"
