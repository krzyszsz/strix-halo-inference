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

MODEL="$MODEL_ROOT/qwen3-coder-30b-a3b-gguf/Qwen3-Coder-30B-A3B-Instruct.Q8_0.gguf"
PORT="${PORT:-8005}"
CTX_SIZE="${CTX_SIZE:-4096}"
GPU_LAYERS="${GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"
OUT_DIR="$REPO_ROOT/llama-cpp-vulkan/out"
CONTAINER="llama-qwen3-coder-30b-test"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$OUT_DIR"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -e MODEL="$MODEL" \
  -e PORT="$PORT" \
  -e CTX_SIZE="$CTX_SIZE" \
  -e GPU_LAYERS="$GPU_LAYERS" \
  -e THREADS="$THREADS" \
  llama-cpp-vulkan:latest

code=""
for _ in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 5
done

if [ "$code" != "200" ]; then
  echo "Server did not become ready on port ${PORT}." >&2
  docker logs --tail 100 "$CONTAINER" || true
  docker rm -f "$CONTAINER" || true
  exit 1
fi

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-gguf",
    "messages": [{"role": "user", "content": "Generate a concise SQL schema for a todo app."}],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 256
  }' > "$OUT_DIR/qwen3_coder_30b_q8.json"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "Saved response to $OUT_DIR/qwen3_coder_30b_q8.json"
