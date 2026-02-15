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
if [ -z "$MODEL_PATH" ]; then
  echo "MODEL_PATH (or MODEL) is required." >&2
  exit 2
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

PORT="${PORT:-8003}"
CTX_SIZE="${CTX_SIZE:-2048}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
PROMPT_MODE="${PROMPT_MODE:-text}"
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-999}"
CONTAINER="${CONTAINER:-llama-ctx-probe-${PORT}}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/llama-cpp-vulkan/out/ctx_probe_${CTX_SIZE}.json}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1200}"
READY_ATTEMPTS="${READY_ATTEMPTS:-180}"
READY_SLEEP_SECS="${READY_SLEEP_SECS:-5}"
READY_LOG_EVERY="${READY_LOG_EVERY:-6}"
REQUEST_HEARTBEAT_SECS="${REQUEST_HEARTBEAT_SECS:-30}"

mkdir -p "$(dirname "$OUT_JSON")"

REQ_JSON="$(mktemp "${TMPDIR:-/tmp}/llama_ctx_req.XXXXXX.json")"
RESP_JSON="$(mktemp "${TMPDIR:-/tmp}/llama_ctx_resp.XXXXXX.json")"
trap 'rm -f "$REQ_JSON" "$RESP_JSON"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

if [ "${PROMPT_MODE}" = "coding" ]; then
  PROMPT_TEXT="${PROMPT_TEXT:-Implement a thread-safe C# utility that computes median and percentile values for a double[] input, then explain algorithmic complexity in 6 bullet points.}"
else
  PROMPT_TEXT="${PROMPT_TEXT:-Explain how to benchmark a local LLM server reproducibly. Provide 8 concise bullet points with practical checks.}"
fi

PROMPT_TEXT="$PROMPT_TEXT" MAX_TOKENS="$MAX_TOKENS" python - <<'PY' > "$REQ_JSON"
import json
import os

payload = {
    "model": "local-gguf",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": os.environ["PROMPT_TEXT"]},
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}
print(json.dumps(payload))
PY

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
log "Starting container=${CONTAINER} ctx=${CTX_SIZE} max_tokens=${MAX_TOKENS} mem=${MEM_LIMIT} swap=${MEMORY_SWAP} reservation=${MEM_RESERVATION}"
cid="$(docker run -d --name "$CONTAINER" \
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
  llama-cpp-vulkan:latest)"
log "Container started id=${cid}"

code=""
for i in $(seq 1 "$READY_ATTEMPTS"); do
  code="$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" || true)"
  if [ "$code" = "200" ]; then
    break
  fi
  if [ $((i % READY_LOG_EVERY)) -eq 0 ]; then
    log "Waiting for server readiness on port=${PORT} attempt=${i}/${READY_ATTEMPTS} http=${code:-none}"
  fi
  sleep "$READY_SLEEP_SECS"
done

if [ "$code" != "200" ]; then
  echo "Server did not become ready on port ${PORT} (ctx=${CTX_SIZE})." >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi

log "Server ready on port=${PORT}; submitting chat request (curl_max_time=${CURL_MAX_TIME}s)"

set +e
curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$REQ_JSON" > "$RESP_JSON" &
curl_pid="$!"
req_start="$(date +%s)"
while kill -0 "$curl_pid" 2>/dev/null; do
  sleep "$REQUEST_HEARTBEAT_SECS"
  now="$(date +%s)"
  elapsed=$((now - req_start))
  log "Chat request still running (elapsed=${elapsed}s ctx=${CTX_SIZE})"
done
wait "$curl_pid"
curl_status=$?
set -e

if [ "$curl_status" -ne 0 ]; then
  echo "Chat request transport error (ctx=${CTX_SIZE})." >&2
  docker logs --tail 200 "$CONTAINER" || true
  exit 1
fi
log "Chat request completed successfully"

RESP_JSON="$RESP_JSON" OUT_JSON="$OUT_JSON" python - <<'PY'
import json
import os
import pathlib
import sys

resp_path = pathlib.Path(os.environ["RESP_JSON"])
resp = json.loads(resp_path.read_text())

if isinstance(resp, dict) and resp.get("error"):
    print(f"Model returned error: {resp['error']}", file=sys.stderr)
    sys.exit(1)

choices = resp.get("choices")
if not choices:
    print(f"Missing choices in response: {resp}", file=sys.stderr)
    sys.exit(1)

msg = choices[0].get("message", {})
content = msg.get("content", "")
if not str(content).strip():
    print(f"Empty content in response: {resp}", file=sys.stderr)
    sys.exit(1)

out_path = pathlib.Path(os.environ["OUT_JSON"])
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2))
print(f"{out_path} bytes={out_path.stat().st_size}")
PY

if [ -f "$OUT_JSON" ] && [ ! -w "$OUT_JSON" ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_JSON" || true
fi

log "Probe completed and artifact saved: $OUT_JSON"
echo "CTX probe success: model=${MODEL_PATH} ctx=${CTX_SIZE} max_tokens=${MAX_TOKENS} out=${OUT_JSON}"
