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

INPUT_IMAGE="${INPUT_IMAGE:-$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_single_compat_2026-02-11.png}"
MODEL_NAME="${MODEL_NAME:-yolo26n-pose.pt}"
DEVICE="${DEVICE:-cpu}"
CONF="${CONF:-0.25}"
IMGSZ="${IMGSZ:-640}"
PORT_TAG="${PORT_TAG:-vision-yolo-pose}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/vision-detection/out/yolo26_pose_result.json}"
OUT_IMAGE="${OUT_IMAGE:-$REPO_ROOT/vision-detection/out/yolo26_pose_annotated.png}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Input image not found: $INPUT_IMAGE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_IMAGE")"

docker run --rm --name "$PORT_TAG" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  -e YOLO_CONFIG_DIR=/tmp \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host \
  -v "$REPO_ROOT:$REPO_ROOT" \
  -w "$REPO_ROOT" \
  vision-yolo-rocm:latest \
  --task pose \
  --model "$MODEL_NAME" \
  --input "$INPUT_IMAGE" \
  --device "$DEVICE" \
  --conf "$CONF" \
  --imgsz "$IMGSZ" \
  --out-json "$OUT_JSON" \
  --out-image "$OUT_IMAGE"

if [ "$(id -u)" -ne 0 ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_JSON" "$OUT_IMAGE" 2>/dev/null || true
fi
