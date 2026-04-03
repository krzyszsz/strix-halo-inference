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

MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q8_0.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/mmproj-gemma-4-26B-A4B-it-f16.gguf}"
PORT="${PORT:-8152}"
CTX_SIZE="${CTX_SIZE:-32768}"
GPU_LAYERS="${GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"
CONTAINER="${CONTAINER:-llama-gemma4-compaction-${PORT}}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y-%m-%d)}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/gemma4-compaction}"
OUT_JSON="${OUT_JSON:-$OUT_DIR/gemma4_compaction_${RUN_TAG}.json}"
OUT_MD="${OUT_MD:-$OUT_DIR/gemma4_compaction_${RUN_TAG}.md}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-240}"
READY_SLEEP_SECS="${READY_SLEEP_SECS:-5}"
LLAMA_DEVICE="${LLAMA_DEVICE:-Vulkan0}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 2
fi
if [ ! -f "$MMPROJ_PATH" ]; then
  echo "Missing mmproj file: $MMPROJ_PATH" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

EXTRA_ARGS="--mmproj $MMPROJ_PATH --jinja --reasoning-budget 0 --reasoning-format none --no-context-shift"
if [ -n "$LLAMA_DEVICE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --device $LLAMA_DEVICE"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/dri \
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

code=""
for _ in $(seq 1 "$READY_ATTEMPTS"); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/models" || true)"
  if [ "$code" = "200" ]; then
    break
  fi
  sleep "$READY_SLEEP_SECS"
done
if [ "$code" != "200" ]; then
  echo "Server did not become ready on port $PORT." >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi

API_BASE="http://127.0.0.1:${PORT}/v1" \
MODEL_NAME="local-gguf" \
OUT_JSON="$OUT_JSON" \
OUT_MD="$OUT_MD" \
  python3 "$SCRIPT_DIR/gemma4_context_compaction_demo.py"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap - EXIT

echo "Saved:"
echo "  $OUT_JSON"
echo "  $OUT_MD"
