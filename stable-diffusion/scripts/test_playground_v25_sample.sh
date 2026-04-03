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

MODEL_ID="$MODEL_ROOT/playground-v2.5-1024px-aesthetic"
DTYPE="${DTYPE:-float16}"
PORT="${PORT:-8006}"
CONTAINER="playground-v25-test"
OUT_DIR="$REPO_ROOT/stable-diffusion/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/playground_v25_1024_$(date +%F).png}"
RESP_DIR="${RESP_DIR:-$REPO_ROOT/reports/retest_$(date +%F)/post}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
RESP_JSON="$RESP_DIR/playground_v25_response.json"
HTTP_STATUS="$(mktemp "${TMPDIR:-/tmp}/playground_v25_http.XXXXXX")"
trap 'rm -f "$HTTP_STATUS"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

mkdir -p "$OUT_DIR"
mkdir -p "$RESP_DIR"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "Starting stable-diffusion server container:"
echo "  model=$MODEL_ID"
echo "  dtype=$DTYPE"
echo "  port=$PORT"
echo "  mem_limit=$MEM_LIMIT mem_swap=$MEMORY_SWAP mem_reservation=$MEM_RESERVATION oom_score_adj=$OOM_SCORE_ADJ"

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
  -e VAE_SLICING="${VAE_SLICING:-1}" \
  -e VAE_TILING="${VAE_TILING:-1}" \
  -e MODEL_CPU_OFFLOAD="${MODEL_CPU_OFFLOAD:-0}" \
  -e SEQUENTIAL_CPU_OFFLOAD="${SEQUENTIAL_CPU_OFFLOAD:-0}" \
  stable-diffusion-rocm:latest

code=""
echo "Waiting for /health..."
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  if [ $((i % 6)) -eq 0 ]; then
    echo "  still waiting (attempt=${i}/180, last_http=${code:-<none>})"
  fi
  sleep 5
done

if [ "$code" != "200" ]; then
  echo "Server did not become ready on port ${PORT}." >&2
  docker logs --tail 200 "$CONTAINER" || true
  docker rm -f "$CONTAINER" || true
  exit 1
fi

# Model card default guidance for Playground v2.5 is ~3. Keep steps moderate for Strix Halo runtimes.
echo "Requesting generation (1024x1024, steps=20, guidance=3.0). This can take a while..."
REQ_TMP="${RESP_JSON}.tmp"
rm -f "$REQ_TMP" || true
set +e
curl -sS --connect-timeout 10 --max-time "${CURL_MAX_TIME:-3600}" \
  -w "%{http_code}" \
  "http://127.0.0.1:${PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "prompt": "A cinematic photograph of a cozy coffee shop with a friendly robot barista, warm tungsten lighting, shallow depth of field, natural skin texture, realistic reflections",
    "parameters": {
      "num_inference_steps": 20,
      "guidance_scale": 3.0,
      "height": 1024,
      "width": 1024
    }
  }' > "$REQ_TMP" 2>/dev/null &
curl_pid=$!
set -e

req_start_ts="$(date +%s)"
while kill -0 "$curl_pid" >/dev/null 2>&1; do
  sleep 60
  now_ts="$(date +%s)"
  elapsed=$((now_ts - req_start_ts))
  echo "  generation still running... elapsed=${elapsed}s"
done

set +e
wait "$curl_pid"
curl_status=$?
set -e

if [ "$curl_status" -ne 0 ]; then
  echo "Generation request failed (curl exit=${curl_status}). Container logs:" >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi

mv -f "$REQ_TMP" "$RESP_JSON"

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
