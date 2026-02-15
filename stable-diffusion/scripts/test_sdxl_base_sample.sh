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

MODEL_ID="$MODEL_ROOT/sdxl-base-1.0"
DTYPE="${DTYPE:-float32}"
PORT="${PORT:-8001}"
CONTAINER="sdxl-base-test"
OUT_DIR="$REPO_ROOT/stable-diffusion/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/sdxl_base_sample.png}"
RESP_DIR="$REPO_ROOT/reports/retest_$(date +%F)/post"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
RESP_JSON="$RESP_DIR/sdxl_base_response.json"
HTTP_STATUS="$(mktemp "${TMPDIR:-/tmp}/sdxl_base_http.XXXXXX")"
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
  -e PORT="$PORT" \
  stable-diffusion-rocm:latest

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

curl -sS --connect-timeout 10 --max-time "${CURL_MAX_TIME:-900}" \
  -w "%{http_code}" \
  "http://127.0.0.1:${PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "prompt": "A futuristic city skyline at dusk, reflective water, ultra-detailed",
    "parameters": {
      "num_inference_steps": 50,
      "guidance_scale": 5.0,
      "height": 512,
      "width": 512
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
