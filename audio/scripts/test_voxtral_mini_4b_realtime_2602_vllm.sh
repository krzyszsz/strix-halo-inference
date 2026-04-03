#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

if [ "$(id -u)" -ne 0 ]; then
  docker() { sudo docker "$@"; }
else
  docker() { command docker "$@"; }
fi

MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm}"
PORT="${PORT:-8131}"
CONTAINER="${CONTAINER:-voxtral-mini-4b-vllm-probe}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_vllm_summary.json}"
OUT_MODELS_JSON="${OUT_MODELS_JSON:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_vllm_models.json}"
OUT_LOG="${OUT_LOG:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_vllm_server.log}"
OUT_STATE_JSON="${OUT_STATE_JSON:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_vllm_state.json}"

STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-}"
HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-}"
HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
PATCH_SITE="${PATCH_SITE:-0}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$(dirname "$OUT_JSON")"
rm -f "$OUT_MODELS_JSON" "$OUT_LOG" "$OUT_STATE_JSON"
read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
EXTRA_ENV_ARGS=()
if [ -n "$HSA_OVERRIDE_GFX_VERSION" ]; then
  EXTRA_ENV_ARGS+=(-e "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION")
fi
if [ -n "$HSA_ENABLE_SDMA" ]; then
  EXTRA_ENV_ARGS+=(-e "HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA")
fi
if [ -n "$HSA_NO_SCRATCH_RECLAIM" ]; then
  EXTRA_ENV_ARGS+=(-e "HSA_NO_SCRATCH_RECLAIM=$HSA_NO_SCRATCH_RECLAIM")
fi
if [ -n "$HIP_VISIBLE_DEVICES" ]; then
  EXTRA_ENV_ARGS+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
fi

PATCH_MOUNT_ARGS=()
if [ "$PATCH_SITE" = "1" ]; then
  # Allow injecting small runtime monkeypatches (via sitecustomize.py) without rebuilding images.
  PATCH_DIR="$REPO_ROOT/audio/patches"
  if [ ! -f "$PATCH_DIR/sitecustomize.py" ]; then
    echo "PATCH_SITE=1 but missing $PATCH_DIR/sitecustomize.py" >&2
    exit 2
  fi
  PATCH_MOUNT_ARGS+=(-v "$PATCH_DIR:/patches:ro")
  # Ensure /patches is on sys.path so Python picks up /patches/sitecustomize.py automatically.
  PATCH_MOUNT_ARGS+=(-e "PYTHONPATH=/patches:${PYTHONPATH:-}")
fi

start_ts="$(date +%s)"
status="failed"
realtime_route="false"
error_text=""
aborted_reason=""

capture_container_artifacts() {
  # Best-effort: capture logs/state even when we're interrupted (watchdog TERM).
  set +e
  if docker ps -a --format '{{.Names}}' | rg -q "^${CONTAINER}\$"; then
    timeout 3 docker logs "$CONTAINER" > "$OUT_LOG" 2>&1 || true
    docker inspect "$CONTAINER" > "$OUT_STATE_JSON" 2>&1 || true
  fi
  set -e
}

write_summary() {
  OUT_JSON="$OUT_JSON" MODEL_ID="$MODEL_ID" STATUS="$status" ERR="$error_text" \
  ELAPSED="$elapsed" PORT="$PORT" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" REALTIME_ROUTE="$realtime_route" \
  HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" HSA_ENABLE_SDMA="$HSA_ENABLE_SDMA" \
  HSA_NO_SCRATCH_RECLAIM="$HSA_NO_SCRATCH_RECLAIM" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  OUT_MODELS_JSON="$OUT_MODELS_JSON" OUT_LOG="$OUT_LOG" OUT_STATE_JSON="$OUT_STATE_JSON" \
  ABORTED_REASON="$aborted_reason" REPO_ROOT="$REPO_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

out_json = Path(os.environ["OUT_JSON"])
repo_root = Path(os.environ["REPO_ROOT"]).resolve()

def rel_for_report(path_str: str) -> str:
    path = Path(path_str).resolve()
    try:
        return str(path.relative_to(repo_root))
    except Exception:
        return str(path)

summary = {
    "model_id": os.environ["MODEL_ID"],
    "status": os.environ["STATUS"],
    "error": os.environ["ERR"],
    "elapsed_seconds": int(os.environ["ELAPSED"]),
    "port": int(os.environ["PORT"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
    "realtime_route_detected": os.environ["REALTIME_ROUTE"] == "true",
    "rocm_env": {
        "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
        "HSA_ENABLE_SDMA": os.environ.get("HSA_ENABLE_SDMA", ""),
        "HSA_NO_SCRATCH_RECLAIM": os.environ.get("HSA_NO_SCRATCH_RECLAIM", ""),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    },
    "models_endpoint_json": rel_for_report(os.environ["OUT_MODELS_JSON"]),
    "server_log": rel_for_report(os.environ["OUT_LOG"]),
    "container_state_json": rel_for_report(os.environ["OUT_STATE_JSON"]),
    "aborted_reason": os.environ.get("ABORTED_REASON", ""),
}
out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(out_json)
PY
}

cleanup() {
  set +e
  capture_container_artifacts
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap 'aborted_reason=signal; error_text=aborted; status=failed; end_ts="$(date +%s)"; elapsed=$((end_ts - start_ts)); cleanup; write_summary; exit 130' INT TERM
trap cleanup EXIT

# Pulling can take a while (and can be the first time this image is used on a fresh machine),
# so keep output enabled for visibility and for watchdog "activity" detection.
docker pull "$IMAGE"

docker run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --group-add video \
  --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  "${EXTRA_ENV_ARGS[@]}" \
  -v "$HF_ROOT:/root/.cache/huggingface" \
  "${PATCH_MOUNT_ARGS[@]}" \
  --entrypoint vllm \
  "$IMAGE" \
  serve \
  --model "$MODEL_ID" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --compilation_config '{"cudagraph_mode":"PIECEWISE"}' \
  "${EXTRA_ARGS_ARR[@]}" >/dev/null

deadline=$((start_ts + STARTUP_TIMEOUT_SEC))
next_status_print_ts="$start_ts"
while [ "$(date +%s)" -lt "$deadline" ]; do
  now_ts="$(date +%s)"
  if [ "$now_ts" -ge "$next_status_print_ts" ]; then
    elapsed=$((now_ts - start_ts))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for vLLM startup (elapsed=${elapsed}s) http://127.0.0.1:${PORT}/v1/models"
    next_status_print_ts=$((now_ts + 30))
  fi

  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" -o "$OUT_MODELS_JSON"; then
    status="ready"
    break
  fi

  if ! docker ps --format '{{.Names}}' | rg -q "^${CONTAINER}\$"; then
    error_text="container_exited"
    break
  fi

  sleep 5
done

docker logs "$CONTAINER" > "$OUT_LOG" 2>&1 || true
docker inspect "$CONTAINER" > "$OUT_STATE_JSON" 2>&1 || true
if rg -q "/v1/realtime" "$OUT_LOG"; then
  realtime_route="true"
fi

if [ "$status" != "ready" ] && [ -z "$error_text" ]; then
  error_text="startup_timeout"
fi

end_ts="$(date +%s)"
elapsed=$((end_ts - start_ts))
write_summary

[ "$status" = "ready" ]
