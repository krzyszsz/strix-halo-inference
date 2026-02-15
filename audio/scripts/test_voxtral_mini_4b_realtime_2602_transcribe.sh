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

MODEL_ID="${MODEL_ID:-/mnt/hf/models/voxtral-mini-4b-realtime-2602-hf}"
INPUT_AUDIO="${INPUT_AUDIO:-$REPO_ROOT/audio/out/voxtral_4b_test_clip_10s.wav}"
OUT_TXT="${OUT_TXT:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_transcript.txt}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_transcribe_summary.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE_CPU="${FORCE_CPU:-}"

CONTAINER="${CONTAINER:-voxtral-mini-4b-transcribe}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$(dirname "$OUT_TXT")" "$(dirname "$OUT_JSON")"

set +e
docker run --rm --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host \
  -e MODEL_ID="$MODEL_ID" \
  -e INPUT_AUDIO="$INPUT_AUDIO" \
  -e OUT_TXT="$OUT_TXT" \
  -e OUT_JSON="$OUT_JSON" \
  -e MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  -e LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
  -e FORCE_CPU="$FORCE_CPU" \
  -e HF_HOME="$HF_ROOT" \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  -w "$REPO_ROOT" \
  voxtral-rocm:latest \
  python audio/scripts/transcribe_voxtral_mini_4b_realtime_2602.py
status=$?
set -e

if [ "$(id -u)" -ne 0 ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_TXT" "$OUT_JSON" 2>/dev/null || true
fi

echo "$OUT_JSON"
exit "$status"

