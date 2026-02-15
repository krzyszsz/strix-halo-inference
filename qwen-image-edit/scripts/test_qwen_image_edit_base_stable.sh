#!/usr/bin/env bash
set -euo pipefail

# Tuned defaults for the *base* Qwen-Image-Edit model on this host.
# Goal: avoid intermittent ROCm GPU hangs while keeping a 512x512 demo reproducible.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

export MODEL_ID="${MODEL_ID:-$MODEL_ROOT/qwen-image-edit}"

export DTYPE="${DTYPE:-bfloat16}"
export TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
export INCLUDE_NEGATIVE_PROMPT="${INCLUDE_NEGATIVE_PROMPT:-1}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-128}"

# Improves stability for some ROCm attention kernels on this platform.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-1}"

# Memory profile tuned for stability (host can still override these).
export MEM_LIMIT="${MEM_LIMIT:-75g}"
export MEMORY_SWAP="${MEMORY_SWAP:-140g}"
export MEM_RESERVATION="${MEM_RESERVATION:-67g}"
export OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

exec bash "$SCRIPT_DIR/test_qwen_image_edit.sh"

