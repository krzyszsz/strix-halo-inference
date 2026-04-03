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

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/qwen-image-edit}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
PORT="${PORT:-8002}"
CONTAINER="qwen-image-edit-full-test"
OUT_DIR="$REPO_ROOT/qwen-image-edit/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen_image_edit_full.png}"
# Keep a repo-local default so this script works even if you haven't run any other
# model suites yet (and so publish reruns don't depend on unrelated outputs).
INPUT_IMAGE="${INPUT_IMAGE:-$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_a_512_seed1234.png}"
STEPS="${STEPS:-50}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
STRENGTH="${STRENGTH:-0.6}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:- }"
INCLUDE_NEGATIVE_PROMPT="${INCLUDE_NEGATIVE_PROMPT:-1}"
PROMPT="${PROMPT:-Add a small red lighthouse on the distant hill}"
CURL_MAX_TIME="${CURL_MAX_TIME:-600}"
HEALTH_RETRIES="${HEALTH_RETRIES:-180}"
HEALTH_SLEEP="${HEALTH_SLEEP:-5}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
VAE_SLICING="${VAE_SLICING:-1}"
VAE_TILING="${VAE_TILING:-1}"
ATTENTION_SLICING="${ATTENTION_SLICING:-0}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"
ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-0}"
PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-}"
REQ_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_image_edit_req.XXXXXX.json")"
RESP_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_image_edit_full.XXXXXX.json")"
HTTP_STATUS="$(mktemp "${TMPDIR:-/tmp}/qwen_image_edit_http.XXXXXX")"
trap 'rm -f "$REQ_JSON" "$RESP_JSON" "$HTTP_STATUS"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

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
  -e DEVICE="$DEVICE" \
  -e PORT="$PORT" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e VAE_SLICING="$VAE_SLICING" \
  -e VAE_TILING="$VAE_TILING" \
  -e ATTENTION_SLICING="$ATTENTION_SLICING" \
  -e ENABLE_MODEL_CPU_OFFLOAD="$ENABLE_MODEL_CPU_OFFLOAD" \
  -e ENABLE_SEQUENTIAL_CPU_OFFLOAD="$ENABLE_SEQUENTIAL_CPU_OFFLOAD" \
  -e HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" \
  -e PYTORCH_HIP_ALLOC_CONF="$PYTORCH_HIP_ALLOC_CONF" \
  qwen-image-edit-rocm:latest

code=""
echo "Waiting for container health on port ${PORT} (retries=${HEALTH_RETRIES}, sleep=${HEALTH_SLEEP}s)..."
for i in $(seq 1 "$HEALTH_RETRIES"); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  if [ $((i % 12)) -eq 0 ]; then
    echo "Health wait progress: attempt ${i}/${HEALTH_RETRIES} (last_code=${code:-none})"
  fi
  sleep "$HEALTH_SLEEP"
done

if [ "$code" != "200" ]; then
  echo "Server did not become ready on port ${PORT}." >&2
  docker logs --tail 100 "$CONTAINER" || true
  docker rm -f "$CONTAINER" || true
  exit 1
fi

export INPUT_IMAGE MODEL_ID PROMPT STEPS TRUE_CFG_SCALE STRENGTH NEGATIVE_PROMPT INCLUDE_NEGATIVE_PROMPT HEIGHT WIDTH MAX_SEQUENCE_LENGTH NUM_IMAGES_PER_PROMPT
python - <<'PY' > "$REQ_JSON"
import base64
import json
import os

with open(os.environ["INPUT_IMAGE"], "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "model": os.environ.get("MODEL_ID", ""),
    "prompt": os.environ.get("PROMPT", ""),
    "image_b64": image_b64,
    "parameters": {},
}
params = payload["parameters"]
params["num_inference_steps"] = int(float(os.environ.get("STEPS", "50")))
params["true_cfg_scale"] = float(os.environ.get("TRUE_CFG_SCALE", "4.0"))
params["strength"] = float(os.environ.get("STRENGTH", "0.6"))
if os.environ.get("INCLUDE_NEGATIVE_PROMPT", "1") == "1":
    params["negative_prompt"] = os.environ.get("NEGATIVE_PROMPT", " ")
if os.environ.get("HEIGHT"):
    params["height"] = int(os.environ["HEIGHT"])
if os.environ.get("WIDTH"):
    params["width"] = int(os.environ["WIDTH"])
if os.environ.get("MAX_SEQUENCE_LENGTH"):
    params["max_sequence_length"] = int(os.environ["MAX_SEQUENCE_LENGTH"])
if os.environ.get("NUM_IMAGES_PER_PROMPT"):
    params["num_images_per_prompt"] = int(os.environ["NUM_IMAGES_PER_PROMPT"])
print(json.dumps(payload))
PY

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  -w "%{http_code}" \
  "http://127.0.0.1:${PORT}/v1/images/edits" \
  -H "Content-Type: application/json" \
  -d @"$REQ_JSON" > "$RESP_JSON" 2>/dev/null &
curl_pid=$!
while kill -0 "$curl_pid" >/dev/null 2>&1; do
  echo "Waiting for edit request to finish... (pid=$curl_pid)"
  sleep 30
done
wait "$curl_pid" || {
  echo "Edit request failed (transport error). Container logs:" >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
}

tail -c 3 "$RESP_JSON" > "$HTTP_STATUS"
truncate -s -3 "$RESP_JSON"
status="$(cat "$HTTP_STATUS")"
if [ "$status" != "200" ]; then
  echo "Edit request returned HTTP ${status}. Response:" >&2
  cat "$RESP_JSON" >&2 || true
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi

RESP_JSON="$RESP_JSON" OUT_PATH="$OUT_PATH" python - <<'PY'
import base64
import json
import os
import pathlib
import sys

resp = json.load(open(os.environ['RESP_JSON']))
if 'data' not in resp or not resp['data']:
    print(f"Unexpected response payload: {resp}", file=sys.stderr)
    sys.exit(1)
img_b64 = resp['data'][0]['b64_json']
path = pathlib.Path(os.environ['OUT_PATH'])
path.write_bytes(base64.b64decode(img_b64))
print(path, path.stat().st_size)
PY
