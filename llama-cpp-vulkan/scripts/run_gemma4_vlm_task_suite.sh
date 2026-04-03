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

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

MODEL_PATH="${MODEL_PATH:-${MODEL:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q8_0.gguf}}"
MMPROJ_PATH="${MMPROJ_PATH:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/mmproj-gemma-4-26B-A4B-it-f16.gguf}"
DESCRIBE_IMAGE_PATH="${DESCRIBE_IMAGE_PATH:-$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png}"
DETECT_IMAGE_PATH="${DETECT_IMAGE_PATH:-$REPO_ROOT/vision-detection/input/bus.jpg}"

PORT="${PORT:-8151}"
CTX_SIZE="${CTX_SIZE:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-999}"
CONTAINER="${CONTAINER:-llama-gemma4-vlm-suite-${PORT}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/gemma4-task-suite}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y-%m-%d)}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-240}"
READY_SLEEP_SECS="${READY_SLEEP_SECS:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1800}"
LLAMA_DEVICE="${LLAMA_DEVICE:-Vulkan0}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model file: $MODEL_PATH" >&2
  exit 2
fi
if [ ! -f "$MMPROJ_PATH" ]; then
  echo "Missing mmproj file: $MMPROJ_PATH" >&2
  exit 2
fi
if [ ! -f "$DESCRIBE_IMAGE_PATH" ]; then
  echo "Missing describe image file: $DESCRIBE_IMAGE_PATH" >&2
  exit 2
fi
if [ ! -f "$DETECT_IMAGE_PATH" ]; then
  echo "Missing detect image file: $DETECT_IMAGE_PATH" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_TAG="${MODEL_TAG:-${MODEL_BASENAME%.gguf}}"
OUT_PREFIX="${MODEL_TAG}_${RUN_TAG}"

CHAT_OUT="$OUT_DIR/${OUT_PREFIX}_chat.json"
CODING_OUT="$OUT_DIR/${OUT_PREFIX}_coding.json"
VISION_DESCRIBE_OUT="$OUT_DIR/${OUT_PREFIX}_vision_describe.json"
VISION_DETECT_OUT="$OUT_DIR/${OUT_PREFIX}_vision_detect.json"

EXTRA_ARGS="--mmproj $MMPROJ_PATH --jinja --reasoning-budget 0 --reasoning-format none --no-context-shift"
if [ -n "$LLAMA_DEVICE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --device $LLAMA_DEVICE"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

log "Starting Gemma4 server: model=$(basename "$MODEL_PATH") ctx=${CTX_SIZE} gpu_layers=${GPU_LAYERS}"
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
for i in $(seq 1 "$READY_ATTEMPTS"); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/models" || true)"
  if [ "$code" = "200" ]; then
    break
  fi
  if [ $((i % 8)) -eq 0 ]; then
    log "Waiting for readiness: attempt=${i}/${READY_ATTEMPTS} http=${code:-none}"
  fi
  sleep "$READY_SLEEP_SECS"
done
if [ "$code" != "200" ]; then
  echo "Server did not become ready on port $PORT." >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi
log "Server ready on port $PORT"

CHAT_REQ="$(mktemp "${TMPDIR:-/tmp}/gemma4_chat_req.XXXXXX.json")"
CODING_REQ="$(mktemp "${TMPDIR:-/tmp}/gemma4_coding_req.XXXXXX.json")"
VISION_DESCRIBE_REQ="$(mktemp "${TMPDIR:-/tmp}/gemma4_vdesc_req.XXXXXX.json")"
VISION_DETECT_REQ="$(mktemp "${TMPDIR:-/tmp}/gemma4_vdet_req.XXXXXX.json")"
trap 'rm -f "$CHAT_REQ" "$CODING_REQ" "$VISION_DESCRIBE_REQ" "$VISION_DETECT_REQ"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$CHAT_REQ"
import json
import os

payload = {
    "model": "local-gguf",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": "/no_think In 6 bullet points explain what tradeoffs matter when choosing a local MoE model for coding and question-answering on a 96GB unified-memory machine."
        },
    ],
    "temperature": 0.4,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}
print(json.dumps(payload))
PY

MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$CODING_REQ"
import json
import os

payload = {
    "model": "local-gguf",
    "messages": [
        {"role": "system", "content": "You are a senior C# engineer."},
        {
            "role": "user",
            "content": (
                "/no_think Write a thread-safe C# class LongContextCache with: "
                "(1) AddTurn(string role, string text), "
                "(2) Compact(int maxChars) that summarizes old turns into one memory item, "
                "(3) BuildPrompt(int keepRecentTurns). "
                "Include XML docs and unit test skeletons."
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

DESCRIBE_IMAGE_PATH="$DESCRIBE_IMAGE_PATH" MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$VISION_DESCRIBE_REQ"
import base64
import json
import os
from pathlib import Path

path = Path(os.environ["DESCRIBE_IMAGE_PATH"])
suffix = path.suffix.lower()
mime = "image/png" if suffix == ".png" else "image/jpeg"
b64 = base64.b64encode(path.read_bytes()).decode("ascii")
payload = {
    "model": "local-gguf",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "/no_think Describe this image in detail. Then list key visible objects and any uncertainty."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
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

DETECT_IMAGE_PATH="$DETECT_IMAGE_PATH" MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY' > "$VISION_DETECT_REQ"
import base64
import json
import os
from pathlib import Path

path = Path(os.environ["DETECT_IMAGE_PATH"])
suffix = path.suffix.lower()
mime = "image/png" if suffix == ".png" else "image/jpeg"
b64 = base64.b64encode(path.read_bytes()).decode("ascii")
payload = {
    "model": "local-gguf",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "/no_think Detect major objects in this image and return strict JSON only with this schema: {\"objects\":[{\"label\":string,\"confidence\":0..1,\"bbox_xyxy\":[x1,y1,x2,y2],\"coord_space\":\"pixel\"}]}. Use pixel coordinates for a 1000x1000 normalized canvas, no extra text."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ],
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "response_format": {"type": "json_object"},
}
print(json.dumps(payload))
PY

log "Running chat / coding / vision requests"
curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$CHAT_REQ" > "$CHAT_OUT"

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$CODING_REQ" > "$CODING_OUT"

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$VISION_DESCRIBE_REQ" > "$VISION_DESCRIBE_OUT"

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$VISION_DETECT_REQ" > "$VISION_DETECT_OUT"

python3 - "$CHAT_OUT" "$CODING_OUT" "$VISION_DESCRIBE_OUT" "$VISION_DETECT_OUT" <<'PY'
import json
import pathlib
import re
import sys

def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
        if m:
            return m.group(1)
    return text

def parse_first_json(text):
    decoder = json.JSONDecoder()
    # Try direct parse first.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced/trimmed extraction.
    cand = extract_json(text)
    try:
        return json.loads(cand)
    except Exception:
        pass
    # Fallback: locate first parseable JSON object/array in mixed text.
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except Exception:
            continue
    raise json.JSONDecodeError("No parseable JSON found", text, 0)

def validate(path, require_json=False):
    p = pathlib.Path(path)
    obj = json.loads(p.read_text())
    if isinstance(obj, dict) and obj.get("error"):
        raise SystemExit(f"{p}: model error -> {obj['error']}")
    ch = obj.get("choices") if isinstance(obj, dict) else None
    if not ch:
        raise SystemExit(f"{p}: missing choices")
    msg = ch[0].get("message", {})
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    if not text:
        raise SystemExit(f"{p}: empty content")
    if require_json:
        parsed = parse_first_json(text)
        if not isinstance(parsed, dict) or "objects" not in parsed:
            raise SystemExit(f"{p}: detection JSON missing top-level 'objects'")
        for idx, item in enumerate(parsed.get("objects", [])):
            if not isinstance(item, dict):
                raise SystemExit(f"{p}: objects[{idx}] is not an object")
            if "bbox_xyxy" not in item and "bbox_norm" not in item:
                raise SystemExit(f"{p}: objects[{idx}] missing bbox field")

validate(sys.argv[1], require_json=False)
validate(sys.argv[2], require_json=False)
validate(sys.argv[3], require_json=False)
validate(sys.argv[4], require_json=True)
print("validated", len(sys.argv) - 1, "outputs")
PY

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
trap - EXIT
rm -f "$CHAT_REQ" "$CODING_REQ" "$VISION_DESCRIBE_REQ" "$VISION_DETECT_REQ"

echo "Saved:"
echo "  $CHAT_OUT"
echo "  $CODING_OUT"
echo "  $VISION_DESCRIBE_OUT"
echo "  $VISION_DETECT_OUT"
