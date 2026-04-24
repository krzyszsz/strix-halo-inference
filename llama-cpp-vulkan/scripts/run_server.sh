#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

docker() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

MODEL_PATH="${MODEL:-${1:-$MODEL_ROOT/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf}}"
PORT="${PORT:-8003}"
CTX_SIZE="${CTX_SIZE:-2048}"
GPU_LAYERS="${GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
USE_DRI="${USE_DRI:-1}"

DOCKER_DEVICE_ARGS=()
if [ "$USE_DRI" = "1" ]; then
  DOCKER_DEVICE_ARGS+=(--device=/dev/dri)
fi

DOCKER_TTY_ARGS=()
if [ -t 0 ]; then
  DOCKER_TTY_ARGS=(-it)
fi

exec docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  "${DOCKER_DEVICE_ARGS[@]}" \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -e MODEL="$MODEL_PATH" \
  -e PORT="$PORT" \
  -e CTX_SIZE="$CTX_SIZE" \
  -e GPU_LAYERS="$GPU_LAYERS" \
  -e THREADS="$THREADS" \
  -e EXTRA_ARGS="$EXTRA_ARGS" \
  llama-cpp-vulkan:latest
