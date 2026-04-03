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

MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
MMPROJ_PATH="${MMPROJ_PATH:-}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png}"
PORT="${PORT:-8130}"
CTX_SIZE="${CTX_SIZE:-32768}"
MAX_TOKENS="${MAX_TOKENS:-768}"
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-999}"
CONTAINER="${CONTAINER:-llama-qwen35-vlm-suite-${PORT}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/qwen35-task-suite}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-180}"
READY_SLEEP_SECS="${READY_SLEEP_SECS:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1800}"
LLAMA_DEVICE="${LLAMA_DEVICE:-Vulkan0}"
REASONING_BUDGET="${REASONING_BUDGET:-0}"
REASONING_FORMAT="${REASONING_FORMAT:-none}"

if [ -z "$MODEL_PATH" ] || [ -z "$MMPROJ_PATH" ]; then
  echo "MODEL_PATH and MMPROJ_PATH are required." >&2
  echo "Example:" >&2
  echo "  MODEL_PATH=$MODEL_ROOT/qwen3.5-9b-gguf/Qwen3.5-9B-Q8_0.gguf \\" >&2
  echo "  MMPROJ_PATH=$MODEL_ROOT/qwen3.5-9b-gguf/mmproj-F16.gguf \\" >&2
  echo "  bash llama-cpp-vulkan/scripts/run_qwen35_vlm_task_suite.sh" >&2
  exit 2
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 2
fi
if [ ! -f "$MMPROJ_PATH" ]; then
  echo "Missing mmproj file: $MMPROJ_PATH" >&2
  exit 2
fi
if [ ! -f "$IMAGE_PATH" ]; then
  echo "Missing image file: $IMAGE_PATH" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_TAG="${MODEL_TAG:-${MODEL_BASENAME%.gguf}}"

EXTRA_ARGS="--mmproj $MMPROJ_PATH --jinja --reasoning-budget $REASONING_BUDGET --reasoning-format $REASONING_FORMAT"
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

VISION_REQ="$(mktemp "${TMPDIR:-/tmp}/qwen35_vision_req.XXXXXX.json")"
TEXT_REQ="$(mktemp "${TMPDIR:-/tmp}/qwen35_text_req.XXXXXX.json")"
CODE_REQ="$(mktemp "${TMPDIR:-/tmp}/qwen35_code_req.XXXXXX.json")"
trap 'rm -f "$VISION_REQ" "$TEXT_REQ" "$CODE_REQ"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

IMAGE_PATH="$IMAGE_PATH" MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$VISION_REQ"
import base64
import json
import os
from pathlib import Path

img_path = Path(os.environ["IMAGE_PATH"])
b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
payload = {
    "model": "local-gguf",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail and list any visible artifacts."},
                {"type": "text", "text": "/no_think Return only the final answer."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ],
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}
print(json.dumps(payload))
PY

MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$TEXT_REQ"
import json
import os

article = (
    "The Fedora workstation had several ROCm-related regressions across kernel updates. "
    "A scripted harness was used to enforce memory limits, cleanup stale containers, and "
    "capture logs for each model run. The final baseline reduced crashes but some models "
    "still required CPU offload. The team now needs a concise summary with action items."
)
payload = {
    "model": "local-gguf",
    "messages": [
        {"role": "system", "content": "You summarize technical reports clearly."},
        {"role": "user", "content": f"/no_think Summarize this in 6 bullet points and 3 action items:\\n\\n{article}"},
    ],
    "temperature": 0.4,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}
print(json.dumps(payload))
PY

MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$CODE_REQ"
import json
import os

payload = {
    "model": "local-gguf",
    "messages": [
        {"role": "system", "content": "You are a senior C# engineer."},
        {
            "role": "user",
            "content": (
                "/no_think Create a thread-safe C# method Optimize(double[] input) that validates input, "
                "runs bounded parallel evaluation, and returns the best score. Include XML docs and a short complexity note."
            ),
        },
    ],
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}
print(json.dumps(payload))
PY

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$VISION_REQ" > "$OUT_DIR/${MODEL_TAG}_vision.json"

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$TEXT_REQ" > "$OUT_DIR/${MODEL_TAG}_summarization.json"

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$CODE_REQ" > "$OUT_DIR/${MODEL_TAG}_coding.json"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap - EXIT
rm -f "$VISION_REQ" "$TEXT_REQ" "$CODE_REQ"

echo "Saved:"
echo "  $OUT_DIR/${MODEL_TAG}_vision.json"
echo "  $OUT_DIR/${MODEL_TAG}_summarization.json"
echo "  $OUT_DIR/${MODEL_TAG}_coding.json"
