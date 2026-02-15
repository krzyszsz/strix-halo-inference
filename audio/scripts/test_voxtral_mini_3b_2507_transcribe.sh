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

MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-3B-2507}"
INPUT_AUDIO="${INPUT_AUDIO:-$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav}"
if [ ! -f "$INPUT_AUDIO" ]; then
  INPUT_AUDIO="$REPO_ROOT/audio/out/podcast_kokoro.wav"
fi

CLIP_SECONDS="${CLIP_SECONDS:-30}"
OUT_TEXT="${OUT_TEXT:-$REPO_ROOT/audio/out/voxtral_mini_3b_2507_transcript.txt}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/audio/out/voxtral_mini_3b_2507_summary.json}"
CONTAINER="${CONTAINER:-voxtral-mini-3b-transcribe-test}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

if [ ! -f "$INPUT_AUDIO" ]; then
  echo "Input audio file not found. Expected one of:" >&2
  echo "  $REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav" >&2
  echo "  $REPO_ROOT/audio/out/podcast_kokoro.wav" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_TEXT")" "$(dirname "$OUT_JSON")"

docker run --rm --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host \
  -e HF_HOME="$HF_ROOT" \
  -e MODEL_ID="$MODEL_ID" \
  -e INPUT_AUDIO="$INPUT_AUDIO" \
  -e CLIP_SECONDS="$CLIP_SECONDS" \
  -e OUT_TEXT="$OUT_TEXT" \
  -e OUT_JSON="$OUT_JSON" \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  -w "$REPO_ROOT" \
  voxtral-rocm:latest \
  python audio/scripts/run_voxtral_mini_3b_2507_transcribe.py

if [ "$(id -u)" -ne 0 ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_TEXT" "$OUT_JSON" "${OUT_TEXT%.txt}.clip.wav" 2>/dev/null || true
fi

echo "$OUT_JSON"
echo "$OUT_TEXT"
