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

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/flux2-klein-4b}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda}"
PORT="${PORT:-8001}"
CONTAINER="${CONTAINER:-flux2-klein-test}"
OUT_DIR="$REPO_ROOT/stable-diffusion/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/flux2_klein_sample.png}"
RESP_DIR="$REPO_ROOT/reports/retest_$(date +%F)/post"
STEPS="${STEPS:-4}"
GUIDANCE="${GUIDANCE:-1.0}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
PROMPT="${PROMPT:-A sleek synthwave cityscape at dusk, neon glow, rain}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
HEALTH_RETRIES="${HEALTH_RETRIES:-120}"
HEALTH_SLEEP="${HEALTH_SLEEP:-5}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
MODEL_CPU_OFFLOAD="${MODEL_CPU_OFFLOAD:-0}"
SEQUENTIAL_CPU_OFFLOAD="${SEQUENTIAL_CPU_OFFLOAD:-0}"
ATTN_SLICING="${ATTN_SLICING:-0}"
VAE_SLICING="${VAE_SLICING:-1}"
VAE_TILING="${VAE_TILING:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-}"
RESP_JSON="$RESP_DIR/flux2_klein_response.json"
HTTP_STATUS="$(mktemp "${TMPDIR:-/tmp}/flux2_klein_http.XXXXXX")"
trap 'rm -f "$HTTP_STATUS"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

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
  --ipc=host --network=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -e MODEL_ID="$MODEL_ID" \
  -e DTYPE="$DTYPE" \
  -e DEVICE="$DEVICE" \
  -e MODEL_CPU_OFFLOAD="$MODEL_CPU_OFFLOAD" \
  -e SEQUENTIAL_CPU_OFFLOAD="$SEQUENTIAL_CPU_OFFLOAD" \
  -e ATTN_SLICING="$ATTN_SLICING" \
  -e VAE_SLICING="$VAE_SLICING" \
  -e VAE_TILING="$VAE_TILING" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e DEVICE_MAP="$DEVICE_MAP" \
  -e MAX_GPU_MEMORY="$MAX_GPU_MEMORY" \
  -e MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" \
  -e PORT="$PORT" \
  stable-diffusion-rocm:latest

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

curl -sS --connect-timeout 10 --max-time "${CURL_MAX_TIME:-900}" \
  -w "%{http_code}" \
  "http://127.0.0.1:${PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "prompt": "'"${PROMPT}"'",
    "parameters": {
      "num_inference_steps": '"${STEPS}"',
      "guidance_scale": '"${GUIDANCE}"',
      "negative_prompt": "'"${NEGATIVE_PROMPT}"'",
      "height": '"${HEIGHT}"',
      "width": '"${WIDTH}"'
    }
  }' > "$RESP_JSON" 2>/dev/null || {
  echo "Generation request failed (transport error). Container logs:" >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
}

# Extract the HTTP status from curl output without polluting JSON payload.
tail -c 3 "$RESP_JSON" > "$HTTP_STATUS"
truncate -s -3 "$RESP_JSON"
status="$(cat "$HTTP_STATUS")"
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
