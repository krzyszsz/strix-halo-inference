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
OUT_JSON="${OUT_JSON:-$REPO_ROOT/audio/out/voxtral_mini_4b_realtime_2602_summary.json}"
CONTAINER="${CONTAINER:-voxtral-mini-4b-realtime-test}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"
FORCE_CPU="${FORCE_CPU:-}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$(dirname "$OUT_JSON")"

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
  -e LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
  -e FORCE_CPU="$FORCE_CPU" \
  -e OUT_JSON="$OUT_JSON" \
  -e HF_HOME="$HF_ROOT" \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  -w "$REPO_ROOT" \
  voxtral-rocm:latest \
  python audio/scripts/probe_voxtral_mini_4b_realtime_2602.py
status=$?
set -e

if [ "$(id -u)" -ne 0 ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_JSON" 2>/dev/null || true
fi

echo "$OUT_JSON"
exit "$status"
