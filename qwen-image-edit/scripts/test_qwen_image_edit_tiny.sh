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

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/qwen-image-edit-tiny-random}"
DTYPE="${DTYPE:-float32}"
PORT="${PORT:-8002}"
CONTAINER="qwen-image-edit-tiny-test"
OUT_DIR="$REPO_ROOT/qwen-image-edit/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen_image_edit_tiny_test.png}"
INPUT_IMAGE="${INPUT_IMAGE:-$REPO_ROOT/qwen-image/out/qwen_image_test.png}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
REQ_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_edit_req.XXXXXX.json")"
RESP_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_edit_resp.XXXXXX.json")"
trap 'rm -f "$REQ_JSON" "$RESP_JSON"' EXIT

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Input image not found: $INPUT_IMAGE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -e MODEL_ID="$MODEL_ID" \
  -e DTYPE="$DTYPE" \
  -e PORT="$PORT" \
  qwen-image-edit-rocm:latest

code=""
for _ in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)
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

MODEL_ID="$MODEL_ID" INPUT_IMAGE="$INPUT_IMAGE" python - <<'PY' > "$REQ_JSON"
import base64, json
import os
img = open(os.environ['INPUT_IMAGE'],'rb').read()
payload = {
    "model": os.environ.get("MODEL_ID", ""),
    "prompt": "Add a red hat to the robot",
    "image_b64": base64.b64encode(img).decode('ascii'),
    "parameters": {
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "strength": 0.8
    }
}
print(json.dumps(payload))
PY

curl -s "http://127.0.0.1:${PORT}/v1/images/edits" \
  -H "Content-Type: application/json" \
  -d @"$REQ_JSON" > "$RESP_JSON"

RESP_JSON="$RESP_JSON" OUT_PATH="$OUT_PATH" python - <<'PY'
import base64, json, os, pathlib
resp = json.load(open(os.environ['RESP_JSON']))
img_b64 = resp['data'][0]['b64_json']
path = pathlib.Path(os.environ['OUT_PATH'])
path.write_bytes(base64.b64decode(img_b64))
print(path, path.stat().st_size)
PY

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
