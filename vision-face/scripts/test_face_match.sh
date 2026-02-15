#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi

REF_DIR="$REPO_ROOT/vision-face/input/ref"
INPUT_DIR="$REPO_ROOT/vision-face/input"
OUT_DIR="$REPO_ROOT/vision-face/out"
mkdir -p "$REF_DIR" "$OUT_DIR"

MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/vision-face}"
DETECTOR_MODEL="${DETECTOR_MODEL:-$MODEL_DIR/ultraface/version-RFB-320.onnx}"
RECOGNIZER_MODEL="${RECOGNIZER_MODEL:-$MODEL_DIR/arcface/arcfaceresnet100-8.onnx}"

if [ ! -f "$DETECTOR_MODEL" ] || [ ! -f "$RECOGNIZER_MODEL" ]; then
  bash "$REPO_ROOT/vision-face/scripts/download_face_models.sh"
fi

# Ensure at least 4 reference images exist. If missing, generate them using a stable local image model.
if [ "$(ls -1 "$REF_DIR"/*.png 2>/dev/null | wc -l)" -lt 4 ]; then
  echo "Generating synthetic reference portraits with FLUX.2-dev-bnb4 (4 images, ~3m each)..."
  for i in 1 2 3 4; do
    prompt=""
    out="$REF_DIR/person_${i}.png"
    case "$i" in
      1) prompt="studio passport photo headshot of a fictional adult woman with short platinum blonde hair and freckles, plain light gray background, photorealistic, sharp focus, soft lighting, not resembling any real person, no text" ;;
      2) prompt="studio passport photo headshot of a fictional adult man with black curly hair, full beard, and glasses, plain light gray background, photorealistic, sharp focus, soft lighting, not resembling any real person, no text" ;;
      3) prompt="studio passport photo headshot of a fictional adult woman with long auburn hair and blue eyes, plain light gray background, photorealistic, sharp focus, soft lighting, not resembling any real person, no text" ;;
      4) prompt="studio passport photo headshot of a fictional adult man with shaved head and a mustache, plain light gray background, photorealistic, sharp focus, soft lighting, not resembling any real person, no text" ;;
    esac

    $SUDO docker run --rm \
      --memory="${MEM_LIMIT:-75g}" \
      --memory-swap="${MEMORY_SWAP:-75g}" \
      --memory-reservation="${MEM_RESERVATION:-67g}" \
      --oom-score-adj="${OOM_SCORE_ADJ:-500}" \
      --device=/dev/kfd \
      --device=/dev/dri \
      --security-opt label=disable \
      --ipc=host --network=host \
      -v "$HF_ROOT:$HF_ROOT" \
      -v "$REF_DIR:/out" \
      -v "$REPO_ROOT/stable-diffusion/scripts/flux2_dev_bnb4_probe.py:/tmp/flux2_dev_bnb4_probe.py:ro" \
      -e HF_HOME="$HF_ROOT" \
      -e HF_HUB_ENABLE_HF_TRANSFER=0 \
      -e MODEL_ID="$MODEL_ROOT/flux2-dev-bnb4" \
      -e OUT_PATH="/out/$(basename "$out")" \
      -e PROMPT="$prompt" \
      -e HEIGHT=512 -e WIDTH=512 -e STEPS=4 -e GUIDANCE=3.0 \
      -e MODEL_CPU_OFFLOAD=1 \
      -e USE_REMOTE_TEXT_ENCODER=0 \
      -e MAX_SEQUENCE_LENGTH=128 \
      -e DTYPE=bfloat16 \
      -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      --entrypoint python stable-diffusion-rocm:latest -u /tmp/flux2_dev_bnb4_probe.py
  done
fi

GROUP_IMAGE="$INPUT_DIR/group.png"
EXPECTED_JSON="$INPUT_DIR/group_expected.json"

OUT_JSON="$OUT_DIR/face_match_results.json"
OUT_IMAGE="$OUT_DIR/face_match_annotated.png"
OUT_SUMMARY="$OUT_DIR/face_match_summary.json"

echo "Running face match demo..."
$SUDO docker run --rm \
  --memory="${MEM_LIMIT:-75g}" \
  --memory-swap="${MEMORY_SWAP:-75g}" \
  --memory-reservation="${MEM_RESERVATION:-67g}" \
  --oom-score-adj="${OOM_SCORE_ADJ:-500}" \
  --security-opt label=disable \
  -v "$MODEL_DIR:/models:ro" \
  -v "$INPUT_DIR:/input" \
  -v "$OUT_DIR:/out" \
  -e DETECTOR_MODEL="/models/ultraface/$(basename "$DETECTOR_MODEL")" \
  -e RECOGNIZER_MODEL="/models/arcface/$(basename "$RECOGNIZER_MODEL")" \
  -e REF_DIR="/input/ref" \
  -e QUERY_IMAGE="/input/$(basename "$GROUP_IMAGE")" \
  -e EXPECTED_JSON="/input/$(basename "$EXPECTED_JSON")" \
  -e OUT_JSON="/out/$(basename "$OUT_JSON")" \
  -e OUT_IMAGE="/out/$(basename "$OUT_IMAGE")" \
  -e OUT_SUMMARY="/out/$(basename "$OUT_SUMMARY")" \
  -e MAKE_SYNTH_COLLAGE=1 \
  -e THREADS="${THREADS:-8}" \
  vision-face-onnx:latest

echo "Wrote: $OUT_JSON"
echo "Wrote: $OUT_IMAGE"
echo "Wrote: $OUT_SUMMARY"
