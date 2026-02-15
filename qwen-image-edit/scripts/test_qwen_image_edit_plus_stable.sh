#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

MODE="${MODE:-single}"  # single|multi
MODEL_ID="${MODEL_ID:-$MODEL_ROOT/qwen-image-edit-2511}"

# Tuned defaults for Qwen-Image-Edit Plus models on this host.
export DTYPE="${DTYPE:-bfloat16}"
export TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-1.0}"
export INCLUDE_NEGATIVE_PROMPT="${INCLUDE_NEGATIVE_PROMPT:-0}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-128}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-1}"

# Memory profile tuned for stability (host can still override these).
export MEM_LIMIT="${MEM_LIMIT:-75g}"
export MEMORY_SWAP="${MEMORY_SWAP:-140g}"
export MEM_RESERVATION="${MEM_RESERVATION:-67g}"
export OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

case "$MODE" in
  single)
    exec env MODEL_ID="$MODEL_ID" bash "$SCRIPT_DIR/test_qwen_image_edit.sh"
    ;;
  multi)
    exec env MODEL_ID="$MODEL_ID" bash "$SCRIPT_DIR/test_qwen_image_edit_multi_input.sh"
    ;;
  *)
    echo "Unsupported MODE='$MODE' (expected: single|multi)" >&2
    exit 2
    ;;
esac
