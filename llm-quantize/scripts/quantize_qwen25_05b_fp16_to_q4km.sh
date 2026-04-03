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
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/retest_${STAMP}_quantize}"
mkdir -p "$OUT_DIR"

IMAGE="${IMAGE:-llama-cpp-tools:latest}"
FORCE_BUILD="${FORCE_BUILD:-0}"

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_DIR="$MODEL_ROOT/qwen2.5-0.5b-instruct-gguf"
FP16_FILE="qwen2.5-0.5b-instruct-fp16.gguf"
FP16_PATH="$MODEL_DIR/$FP16_FILE"

Q4_LOCAL_FILE="qwen2.5-0.5b-instruct-q4_k_m-local.gguf"
Q4_LOCAL_PATH="$MODEL_DIR/$Q4_LOCAL_FILE"

PROMPT='Write a 4-line poem about Vulkan on AMD Strix Halo.'
N_PREDICT="${N_PREDICT:-128}"
THREADS="${THREADS:-8}"
SEED="${SEED:-42}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$MODEL_DIR"

echo "[quantize] image=$IMAGE"
echo "[quantize] model_id=$MODEL_ID"
echo "[quantize] fp16=$FP16_PATH"
echo "[quantize] q4_local=$Q4_LOCAL_PATH"
echo "[quantize] out_dir=$OUT_DIR"

if [ "$FORCE_BUILD" = "1" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[quantize] building $IMAGE (FORCE_BUILD=$FORCE_BUILD)"
  DOCKER_BUILDKIT=1 docker build -t "$IMAGE" "$REPO_ROOT/llm-quantize"
fi

TOKEN=""
if [ -f "$HF_TOKEN_FILE" ]; then
  # Do not echo token.
  TOKEN="$(cat "$HF_TOKEN_FILE" | tr -d '\n' || true)"
fi

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
  -e OUT_DIR="$OUT_DIR" \
  -e HF_HOME="$MODEL_ROOT/.cache/huggingface" \
  -e TRANSFORMERS_CACHE="$MODEL_ROOT/.cache/huggingface/transformers" \
  -e HUGGINGFACE_HUB_CACHE="$MODEL_ROOT/.cache/huggingface/hub" \
  -e HF_TOKEN_FILE="$HF_TOKEN_FILE" \
  -e MODEL_ROOT="$MODEL_ROOT" \
  -e MODEL_ID="$MODEL_ID" \
  -e MODEL_DIR="$MODEL_DIR" \
  -e FP16_FILE="$FP16_FILE" \
  -e FP16_PATH="$FP16_PATH" \
  -e Q4_LOCAL_PATH="$Q4_LOCAL_PATH" \
  -e N_PREDICT="$N_PREDICT" \
  -e THREADS="$THREADS" \
  -e SEED="$SEED" \
  -e PROMPT="$PROMPT" \
  -e HUGGINGFACE_HUB_TOKEN="${TOKEN:-}" \
  "$IMAGE" \
  'set -euo pipefail
   mkdir -p "$MODEL_DIR"
   mkdir -p "$MODEL_ROOT/.cache/huggingface"
   if [ -f "$FP16_PATH" ] && [ "$(stat -c%s "$FP16_PATH" 2>/dev/null || echo 0)" -gt 0 ]; then
     echo "[container] fp16 already present, skipping download: $FP16_PATH"
   else
     echo "[container] downloading: $MODEL_ID -> $FP16_PATH"
     huggingface-cli download "$MODEL_ID" \
       --include "$FP16_FILE" \
       --local-dir "$MODEL_DIR" \
       --local-dir-use-symlinks False
   fi
   test -f "$FP16_PATH"
   echo "[container] fp16_bytes=$(stat -c%s "$FP16_PATH")"
   echo "[container] quantizing: FP16 -> Q4_K_M"
   /usr/local/bin/llama-quantize "$FP16_PATH" "$Q4_LOCAL_PATH" Q4_K_M
   test -f "$Q4_LOCAL_PATH"
   echo "[container] q4_bytes=$(stat -c%s "$Q4_LOCAL_PATH")"
   echo "[container] inference (fp16)"
   /usr/local/bin/llama-cli -m "$FP16_PATH" -p "$PROMPT" -n "$N_PREDICT" -t "$THREADS" --seed "$SEED" --temp 0.7 --top-p 0.9 --no-display-prompt --single-turn \
     | tee "$OUT_DIR/fp16_response.txt"
   echo "[container] inference (q4)"
   /usr/local/bin/llama-cli -m "$Q4_LOCAL_PATH" -p "$PROMPT" -n "$N_PREDICT" -t "$THREADS" --seed "$SEED" --temp 0.7 --top-p 0.9 --no-display-prompt --single-turn \
     | tee "$OUT_DIR/q4_response.txt"
   python3 - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
out_dir = Path(os.environ["OUT_DIR"])
fp16_path = Path(os.environ["FP16_PATH"])
q4_path = Path(os.environ["Q4_LOCAL_PATH"])

out = {
  "model_id": os.environ["MODEL_ID"],
  "fp16_path": str(fp16_path),
  "q4_path": str(q4_path),
  "prompt": os.environ["PROMPT"],
  "n_predict": int(os.environ["N_PREDICT"]),
  "threads": int(os.environ["THREADS"]),
  "seed": int(os.environ["SEED"]),
  "fp16_bytes": fp16_path.stat().st_size,
  "q4_bytes": q4_path.stat().st_size,
  "fp16_response": (out_dir / "fp16_response.txt").read_text(encoding="utf-8", errors="replace").strip(),
  "q4_response": (out_dir / "q4_response.txt").read_text(encoding="utf-8", errors="replace").strip(),
}

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "quantize_outputs.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("[container] wrote", out_dir / "quantize_outputs.json")
PY'

echo "[quantize] done"
