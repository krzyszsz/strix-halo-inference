#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/qwen3.6-27b-gguf/Qwen-Qwen3.6-27B-Q4_K_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-$MODEL_ROOT/qwen3.6-27b-gguf/mmproj-Qwen-Qwen3.6-27B-BF16.gguf}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png}"
PORT="${PORT:-8161}"
MODEL_TAG="${MODEL_TAG:-qwen36_27b_q4km_$(date -u +%Y_%m_%d)}"
CTX_SIZE="${CTX_SIZE:-32768}"
MAX_TOKENS="${MAX_TOKENS:-768}"
GPU_LAYERS="${GPU_LAYERS:-999}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/qwen36-task-suite}"

export MODEL_PATH MMPROJ_PATH IMAGE_PATH PORT MODEL_TAG CTX_SIZE MAX_TOKENS GPU_LAYERS MEM_LIMIT MEMORY_SWAP MEM_RESERVATION OUT_DIR

exec "$REPO_ROOT/llama-cpp-vulkan/scripts/run_qwen35_vlm_task_suite.sh"
