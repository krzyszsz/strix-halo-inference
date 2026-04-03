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

MODEL_ID="$MODEL_ROOT/qwen2.5-vl-7b-instruct"
DTYPE="${DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
PORT="${PORT:-8005}"
CONTAINER="qwen-vl-test"
OUT_DIR="$REPO_ROOT/qwen-vl/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen_vl_describe.txt}"
INPUT_IMAGE="${INPUT_IMAGE:-$REPO_ROOT/qwen-vl/input/qwen_image_full_256.png}"
MIN_PIXELS="${MIN_PIXELS:-}"
MAX_PIXELS="${MAX_PIXELS:-}"
PROMPT="${PROMPT:-Describe the image and list any AI artifacts or mistakes you see.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
REQ_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_vl_req.XXXXXX.json")"
RESP_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_vl_resp.XXXXXX.json")"
trap 'rm -f "$REQ_JSON" "$RESP_JSON"' EXIT

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Input image not found: $INPUT_IMAGE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [ "${FORCE_CPU:-0}" = "1" ]; then
  attempts=(
    "float32 eager cpu"
  )
else
  attempts=(
    "float16 eager auto"
    "float32 eager auto"
    "float32 eager cpu"
  )
fi

success="false"
for attempt in "${attempts[@]}"; do
  read -r attempt_dtype attempt_attn attempt_device <<<"$attempt"
  echo "Attempt: DTYPE=${attempt_dtype} ATTN_IMPL=${attempt_attn} DEVICE_MAP=${attempt_device}"

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  extra_env=()
  if [ -n "$MIN_PIXELS" ]; then
    extra_env+=(-e "MIN_PIXELS=$MIN_PIXELS")
  fi
  if [ -n "$MAX_PIXELS" ]; then
    extra_env+=(-e "MAX_PIXELS=$MAX_PIXELS")
  fi

  docker run -d --name "$CONTAINER" \
    --memory="$MEM_LIMIT" \
    --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host --network=host \
    -v "$HF_ROOT:$HF_ROOT" \
    -e MODEL_ID="$MODEL_ID" \
    -e DTYPE="$attempt_dtype" \
    -e ATTN_IMPL="$attempt_attn" \
    -e DEVICE_MAP="$attempt_device" \
    -e PORT="$PORT" \
    "${extra_env[@]}" \
    qwen-vl-rocm:latest

  code=""
  for _ in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)
    if [ "$code" = "200" ]; then
      break
    fi
    sleep 5
    done

  if [ "$code" != "200" ]; then
    echo "Server did not become ready on port ${PORT}." >&2
    docker logs --tail 200 "$CONTAINER" || true
    docker rm -f "$CONTAINER" || true
    continue
  fi

export IMAGE_PATH="$INPUT_IMAGE"
python - <<'PY' > "$REQ_JSON"
import base64
import json
import os

with open(os.environ["IMAGE_PATH"], "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "prompt": os.environ.get("PROMPT", ""),
    "image_b64": image_b64,
    "parameters": {
        "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "256")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.2")),
        "top_p": float(os.environ.get("TOP_P", "0.8")),
        "top_k": int(os.environ.get("TOP_K", "20")),
    },
}
print(json.dumps(payload))
PY

CURL_MAX_TIME="${CURL_MAX_TIME:-900}"
if ! curl -s --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/vision/describe" \
  -H "Content-Type: application/json" \
  -d @"$REQ_JSON" > "$RESP_JSON"; then
  echo "Request failed; dumping logs." >&2
  docker logs --tail 200 "$CONTAINER" || true
  docker rm -f "$CONTAINER" || true
  continue
fi

RESP_JSON="$RESP_JSON" OUT_PATH="$OUT_PATH" python - <<'PY'
import json
import os
import pathlib

resp = json.load(open(os.environ['RESP_JSON']))
text = resp.get('text', '')
path = pathlib.Path(os.environ['OUT_PATH'])
path.write_text(text)
print(path, path.stat().st_size)
PY
success="true"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
break
done

if [ "$success" != "true" ]; then
  echo "All attempts failed." >&2
  exit 1
fi
