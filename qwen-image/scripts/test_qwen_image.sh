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

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/qwen-image-2512}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-8000}"
CONTAINER="qwen-image-full-test"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-qwen-image-rocm:latest}"
OUT_DIR="$REPO_ROOT/qwen-image/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen_image_full.png}"
RESP_DIR="$REPO_ROOT/reports/retest_$(date +%F)/post"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEPS="${STEPS:-30}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
PROMPT="${PROMPT:-A cinematic robot barista pouring latte art, warm lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:- }"
SEED="${SEED:-}"
CURL_MAX_TIME="${CURL_MAX_TIME:-600}"
HEALTH_RETRIES="${HEALTH_RETRIES:-180}"
HEALTH_SLEEP="${HEALTH_SLEEP:-5}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
VAE_SLICING="${VAE_SLICING:-1}"
VAE_TILING="${VAE_TILING:-1}"
ATTENTION_SLICING="${ATTENTION_SLICING:-0}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-0}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-}"
RESP_JSON="$RESP_DIR/qwen_image_full_response.json"
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

mkdir -p "$OUT_DIR"
mkdir -p "$RESP_DIR"

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
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e VAE_SLICING="$VAE_SLICING" \
  -e VAE_TILING="$VAE_TILING" \
  -e ATTENTION_SLICING="$ATTENTION_SLICING" \
  -e ENABLE_MODEL_CPU_OFFLOAD="$ENABLE_MODEL_CPU_OFFLOAD" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" \
  "$CONTAINER_IMAGE"

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

seed_json=""
if [ -n "$SEED" ]; then
  if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be an integer (got: '$SEED')" >&2
    exit 2
  fi
  seed_json=",\"seed\": ${SEED}"
fi

max_sequence_json=""
if [ -n "$MAX_SEQUENCE_LENGTH" ]; then
  if ! [[ "$MAX_SEQUENCE_LENGTH" =~ ^[0-9]+$ ]]; then
    echo "MAX_SEQUENCE_LENGTH must be an integer (got: '$MAX_SEQUENCE_LENGTH')" >&2
    exit 2
  fi
  max_sequence_json=",\"max_sequence_length\": ${MAX_SEQUENCE_LENGTH}"
fi

status="$(curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  -o "$RESP_JSON" \
  -w "%{http_code}" \
  "http://127.0.0.1:${PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d "{\
    \"model\": \"${MODEL_ID}\",\
    \"prompt\": \"${PROMPT}\",\
    \"parameters\": {\
      \"num_inference_steps\": ${STEPS},\
      \"true_cfg_scale\": ${TRUE_CFG_SCALE},\
      \"negative_prompt\": \"${NEGATIVE_PROMPT}\",\
      \"height\": ${HEIGHT},\
      \"width\": ${WIDTH}${seed_json}${max_sequence_json}\
    }\
  }" 2>/dev/null)" || {
  echo "Generation request failed (transport error). Container logs:" >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
}
if [ "$status" != "200" ]; then
  echo "Generation request returned HTTP ${status}. Response saved at ${RESP_JSON}" >&2
  cat "$RESP_JSON" >&2 || true
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi

RESP_JSON="$RESP_JSON" OUT_PATH="$OUT_PATH" python - <<'PY'
import base64, json, os, pathlib, sys
resp = json.load(open(os.environ['RESP_JSON']))
if 'data' not in resp or not resp['data']:
    print(f"Unexpected response payload: {resp}", file=sys.stderr)
    sys.exit(1)
img_b64 = resp['data'][0]['b64_json']
path = pathlib.Path(os.environ['OUT_PATH'])
path.write_bytes(base64.b64decode(img_b64))
print(path, path.stat().st_size)
PY
