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

STAMP="${STAMP:-$(date -u +%F)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/retest_${STAMP}_finetune}"
mkdir -p "$OUT_DIR"

IMAGE="${IMAGE:-llm-finetune:latest}"
FORCE_BUILD="${FORCE_BUILD:-0}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

if [ "$FORCE_BUILD" = "1" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[finetune] building $IMAGE (FORCE_BUILD=$FORCE_BUILD)"
  DOCKER_BUILDKIT=1 docker build -t "$IMAGE" "$REPO_ROOT/llm-finetune"
fi

echo "[finetune] image=$IMAGE"
echo "[finetune] out_dir=$OUT_DIR"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --security-opt label=disable \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  -e REPO_ROOT="$REPO_ROOT" \
  -e MODEL_ROOT="$MODEL_ROOT" \
  -e EVIDENCE_DIR="$OUT_DIR" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME="$MODEL_ROOT/.cache/huggingface" \
  -e TRANSFORMERS_CACHE="$MODEL_ROOT/.cache/huggingface/transformers" \
  -e HUGGINGFACE_HUB_CACHE="$MODEL_ROOT/.cache/huggingface/hub" \
  "$IMAGE" \
  "python $REPO_ROOT/llm-finetune/scripts/finetune_smollm2_135m_lora_demo.py"

echo "[finetune] done"
