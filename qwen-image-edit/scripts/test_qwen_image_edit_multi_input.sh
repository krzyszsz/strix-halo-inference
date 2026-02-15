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

MODEL_ID="${MODEL_ID:-Qwen/Qwen-Image-Edit-2509}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
PORT="${PORT:-8012}"
CONTAINER="${CONTAINER:-qwen-image-edit-multi-test}"
OUT_DIR="$REPO_ROOT/qwen-image-edit/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen_image_edit_multi_compose_512.png}"
INPUT_IMAGE_A="${INPUT_IMAGE_A:-$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png}"
INPUT_IMAGE_B="${INPUT_IMAGE_B:-$REPO_ROOT/stable-diffusion/out/sdxl_base_best_retest.png}"
PROMPT="${PROMPT:-Combine both source images into one coherent scene: keep the teenage person from image 1 and place him naturally into image 2, preserving realistic lighting and anatomy.}"
STEPS="${STEPS:-8}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:- }"
INCLUDE_NEGATIVE_PROMPT="${INCLUDE_NEGATIVE_PROMPT:-1}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-}"
SEED="${SEED:-}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1800}"
HEALTH_RETRIES="${HEALTH_RETRIES:-240}"
HEALTH_SLEEP="${HEALTH_SLEEP:-5}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
VAE_SLICING="${VAE_SLICING:-1}"
VAE_TILING="${VAE_TILING:-1}"
ATTENTION_SLICING="${ATTENTION_SLICING:-1}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-0}"
PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"
ENABLE_SEQUENTIAL_CPU_OFFLOAD="${ENABLE_SEQUENTIAL_CPU_OFFLOAD:-0}"

REQ_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_image_edit_multi_req.XXXXXX.json")"
RESP_JSON="$(mktemp "${TMPDIR:-/tmp}/qwen_image_edit_multi_resp.XXXXXX.json")"
HTTP_STATUS="$(mktemp "${TMPDIR:-/tmp}/qwen_image_edit_multi_http.XXXXXX")"
trap 'rm -f "$REQ_JSON" "$RESP_JSON" "$HTTP_STATUS"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

if [ ! -f "$INPUT_IMAGE_A" ]; then
  echo "Input image A not found: $INPUT_IMAGE_A" >&2
  exit 1
fi
if [ ! -f "$INPUT_IMAGE_B" ]; then
  echo "Input image B not found: $INPUT_IMAGE_B" >&2
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
  -e HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e PYTORCH_HIP_ALLOC_CONF="$PYTORCH_HIP_ALLOC_CONF" \
  -e VAE_SLICING="$VAE_SLICING" \
  -e VAE_TILING="$VAE_TILING" \
  -e ATTENTION_SLICING="$ATTENTION_SLICING" \
  -e ENABLE_MODEL_CPU_OFFLOAD="$ENABLE_MODEL_CPU_OFFLOAD" \
  -e ENABLE_SEQUENTIAL_CPU_OFFLOAD="$ENABLE_SEQUENTIAL_CPU_OFFLOAD" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" \
  qwen-image-edit-rocm:latest

code=""
echo "Waiting for container health on port ${PORT} (retries=${HEALTH_RETRIES}, sleep=${HEALTH_SLEEP}s)..."
for i in $(seq 1 "$HEALTH_RETRIES"); do
  code="$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)"
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
  docker logs --tail 120 "$CONTAINER" || true
  exit 1
fi

export INPUT_IMAGE_A INPUT_IMAGE_B MODEL_ID PROMPT STEPS TRUE_CFG_SCALE NEGATIVE_PROMPT INCLUDE_NEGATIVE_PROMPT HEIGHT WIDTH MAX_SEQUENCE_LENGTH NUM_IMAGES_PER_PROMPT
python - <<'PY' > "$REQ_JSON"
import base64
import json
import os

with open(os.environ["INPUT_IMAGE_A"], "rb") as f:
    image_a_b64 = base64.b64encode(f.read()).decode("ascii")
with open(os.environ["INPUT_IMAGE_B"], "rb") as f:
    image_b_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "model": os.environ["MODEL_ID"],
    "prompt": os.environ["PROMPT"],
    "images_b64": [image_a_b64, image_b_b64],
    "parameters": {
        "num_inference_steps": int(float(os.environ.get("STEPS", "8"))),
        "true_cfg_scale": float(os.environ.get("TRUE_CFG_SCALE", "4.0")),
        "height": int(float(os.environ.get("HEIGHT", "512"))),
        "width": int(float(os.environ.get("WIDTH", "512"))),
    },
}
seed = os.environ.get("SEED")
if seed:
    payload["parameters"]["seed"] = int(seed)
if os.environ.get("INCLUDE_NEGATIVE_PROMPT", "1") == "1":
    payload["parameters"]["negative_prompt"] = os.environ.get("NEGATIVE_PROMPT", " ")
if os.environ.get("MAX_SEQUENCE_LENGTH"):
    payload["parameters"]["max_sequence_length"] = int(os.environ["MAX_SEQUENCE_LENGTH"])
if os.environ.get("NUM_IMAGES_PER_PROMPT"):
    payload["parameters"]["num_images_per_prompt"] = int(os.environ["NUM_IMAGES_PER_PROMPT"])
print(json.dumps(payload))
PY

curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
  -w "%{http_code}" \
  "http://127.0.0.1:${PORT}/v1/images/edits" \
  -H "Content-Type: application/json" \
  -d @"$REQ_JSON" > "$RESP_JSON" 2>/dev/null &
curl_pid=$!

# Keep the harness log moving so the activity watchdog doesn't look "stuck"
# during long requests (curl itself is silent until it finishes).
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

resp = json.load(open(os.environ["RESP_JSON"]))
if "data" not in resp or not resp["data"]:
    print(f"Unexpected response payload: {resp}", file=sys.stderr)
    sys.exit(1)
img_b64 = resp["data"][0]["b64_json"]
out_path = pathlib.Path(os.environ["OUT_PATH"])
out_path.write_bytes(base64.b64decode(img_b64))
print(out_path, out_path.stat().st_size)
PY
