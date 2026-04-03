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

SCENE_NAME="${SCENE_NAME:-south_building}"
SCENE_BASENAME="${SCENE_BASENAME:-${SCENE_NAME}_points}"
SCENE_DIR="${SCENE_DIR:-$REPO_ROOT/reconstruction-3d/data/$SCENE_NAME}"
IMAGE_DIR="${IMAGE_DIR:-$SCENE_DIR/images}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reconstruction-3d/out/$SCENE_NAME}"
OUT_PLY="${OUT_PLY:-$OUT_DIR/${SCENE_BASENAME}.ply}"
OUT_PREVIEW="${OUT_PREVIEW:-$OUT_DIR/${SCENE_BASENAME}_preview.png}"
OUT_JSON="${OUT_JSON:-$OUT_DIR/${SCENE_NAME}_summary.json}"
MODEL_ID="${MODEL_ID:-facebook/VGGT-1B}"
LOAD_RESOLUTION="${LOAD_RESOLUTION:-768}"
MODEL_RESOLUTION="${MODEL_RESOLUTION:-448}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.0}"
MAX_POINTS="${MAX_POINTS:-200000}"
PREVIEW_POINTS="${PREVIEW_POINTS:-60000}"
SEED="${SEED:-42}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
CONTAINER="${CONTAINER:-vggt-reconstruct}"

if [ ! -d "$IMAGE_DIR" ]; then
  echo "Image directory not found: $IMAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

docker run --rm --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e HF_HOME="$HF_ROOT" \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  -w "$REPO_ROOT" \
  vggt-rocm:latest \
  python reconstruction-3d/scripts/reconstruct_scene.py \
    --image-dir "$IMAGE_DIR" \
    --out-ply "$OUT_PLY" \
    --out-preview "$OUT_PREVIEW" \
    --out-json "$OUT_JSON" \
    --model-id "$MODEL_ID" \
    --load-resolution "$LOAD_RESOLUTION" \
    --model-resolution "$MODEL_RESOLUTION" \
    --conf-threshold "$CONF_THRESHOLD" \
    --max-points "$MAX_POINTS" \
    --preview-points "$PREVIEW_POINTS" \
    --seed "$SEED"

if [ "$(id -u)" -ne 0 ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_PLY" "$OUT_PREVIEW" "$OUT_JSON" 2>/dev/null || true
fi

echo "$OUT_JSON"
echo "$OUT_PLY"
echo "$OUT_PREVIEW"
