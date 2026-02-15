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

# This probe runs the lightweight python pipeline directly (no REST server) to avoid
# additional FastAPI overhead and make it easier to add i2i/multi-image init refs.

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/flux2-klein-9b}"
OUT_PATH="${OUT_PATH:-$REPO_ROOT/stable-diffusion/out/flux2_klein_probe.png}"

PROMPT="${PROMPT:-A cinematic photo of a robot barista serving coffee in warm light}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
STEPS="${STEPS:-4}"
GUIDANCE="${GUIDANCE:-1.0}"
SEED="${SEED:-42}"
MODEL_CPU_OFFLOAD="${MODEL_CPU_OFFLOAD:-1}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
TEXT_ENCODER_OUT_LAYERS="${TEXT_ENCODER_OUT_LAYERS:-9,18,27}"
INIT_IMAGE="${INIT_IMAGE:-}" # optional, comma-separated; repo-relative or absolute under $REPO_ROOT

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"

OUT_DIR_HOST="$(cd "$(dirname "$OUT_PATH")" && pwd)"
OUT_BASE="$(basename "$OUT_PATH")"
OUT_PATH_CONTAINER="/out/$OUT_BASE"

REPO_MOUNT="/repo"

map_repo_path() {
  local p="$1"
  if [[ "$p" == "$REPO_ROOT"* ]]; then
    printf "%s%s" "$REPO_MOUNT" "${p#$REPO_ROOT}"
    return
  fi
  if [[ "$p" != /* ]]; then
    printf "%s/%s" "$REPO_MOUNT" "$p"
    return
  fi
  printf "%s" "$p"
}

INIT_IMAGE_CONTAINER=""
if [ -n "$INIT_IMAGE" ]; then
  first=1
  for item in ${INIT_IMAGE//,/ }; do
    mapped="$(map_repo_path "$item")"
    if [ $first -eq 1 ]; then
      INIT_IMAGE_CONTAINER="$mapped"
      first=0
    else
      INIT_IMAGE_CONTAINER="${INIT_IMAGE_CONTAINER},${mapped}"
    fi
  done
fi

mkdir -p "$OUT_DIR_HOST"

docker run --rm \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host --network=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_MOUNT:ro,Z" \
  -v "$OUT_DIR_HOST:/out:Z" \
  -e HF_HOME="$HF_ROOT" \
  -e MODEL_ID="$MODEL_ID" \
  -e OUT_PATH="$OUT_PATH_CONTAINER" \
  -e PROMPT="$PROMPT" \
  -e HEIGHT="$HEIGHT" \
  -e WIDTH="$WIDTH" \
  -e STEPS="$STEPS" \
  -e GUIDANCE="$GUIDANCE" \
  -e SEED="$SEED" \
  -e MODEL_CPU_OFFLOAD="$MODEL_CPU_OFFLOAD" \
  -e MAX_SEQUENCE_LENGTH="$MAX_SEQUENCE_LENGTH" \
  -e TEXT_ENCODER_OUT_LAYERS="$TEXT_ENCODER_OUT_LAYERS" \
  -e INIT_IMAGE="$INIT_IMAGE_CONTAINER" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" \
  --entrypoint python \
  stable-diffusion-rocm:latest \
  -u "$REPO_MOUNT/stable-diffusion/scripts/flux2_klein_probe.py"

if [ -f "$OUT_PATH" ] && [ ! -w "$OUT_PATH" ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_PATH" || true
fi

echo "Saved: $OUT_PATH"
