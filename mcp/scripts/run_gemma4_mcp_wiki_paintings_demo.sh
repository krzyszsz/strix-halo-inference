#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

docker_cmd() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/mmproj-gemma-4-26B-A4B-it-f16.gguf}"
LLM_PORT="${LLM_PORT:-8153}"
LLM_BASE="${LLM_BASE:-http://127.0.0.1:${LLM_PORT}/v1}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-999}"
LLAMA_DEVICE="${LLAMA_DEVICE:-Vulkan0}"
OUT_TAG="${OUT_TAG:-$(date -u +%F)}"
CONTAINER="${CONTAINER:-llama-gemma4-mcp-${LLM_PORT}}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-180}"
READY_SLEEP_SECS="${READY_SLEEP_SECS:-5}"

PLAYWRIGHT_PORT="${PLAYWRIGHT_PORT:-8935}"
SHELL_MCP_PORT="${SHELL_MCP_PORT:-8023}"
RESIZE_IMAGE_TAG="${RESIZE_IMAGE_TAG:-mcp-image-tools:1.0}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 2
fi
if [ ! -f "$MMPROJ_PATH" ]; then
  echo "Missing mmproj file: $MMPROJ_PATH" >&2
  exit 2
fi

EXTRA_ARGS="--mmproj $MMPROJ_PATH --jinja --reasoning-budget 0 --reasoning-format none --no-context-shift"
if [ -n "$LLAMA_DEVICE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --device $LLAMA_DEVICE"
fi

docker_cmd rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap 'docker_cmd rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

echo "Building image tools container: $RESIZE_IMAGE_TAG"
IMAGE_TAG="$RESIZE_IMAGE_TAG" bash "$REPO_ROOT/mcp/scripts/build_image_tools.sh"

echo "Starting Gemma4 server on port $LLM_PORT"
docker_cmd run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -e MODEL="$MODEL_PATH" \
  -e PORT="$LLM_PORT" \
  -e CTX_SIZE="$CTX_SIZE" \
  -e GPU_LAYERS="$GPU_LAYERS" \
  -e THREADS="$THREADS" \
  -e EXTRA_ARGS="$EXTRA_ARGS" \
  llama-cpp-vulkan:latest >/dev/null

code=""
for i in $(seq 1 "$READY_ATTEMPTS"); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${LLM_PORT}/v1/models" || true)"
  if [ "$code" = "200" ]; then
    break
  fi
  sleep "$READY_SLEEP_SECS"
done
if [ "$code" != "200" ]; then
  echo "Gemma4 server did not become ready on port $LLM_PORT" >&2
  docker_cmd logs --tail 200 "$CONTAINER" || true
  exit 1
fi

export LLM_BASE
export LLM_MODEL="${LLM_MODEL:-local-gguf}"
export PLAYWRIGHT_PORT
export SHELL_MCP_PORT
export OUT_TAG
export RESIZE_IMAGE_TAG

python3 "$REPO_ROOT/mcp/scripts/gemma4_playwright_wiki_paintings_demo.py"

docker_cmd rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap - EXIT

