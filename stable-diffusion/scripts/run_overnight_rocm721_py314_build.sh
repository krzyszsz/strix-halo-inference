#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TAG="${TAG:-stable-diffusion-rocm:rocm7.2.1-py3.14-pt2.10.0-20260331}"
ROCM_BASE="${ROCM_BASE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.14_pytorch_2.10.0}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/stable-diffusion/out/overnight_rocm721_py314_20260331}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/build_and_test.log}"
REPORT_PATH="${REPORT_PATH:-$RUN_DIR/report.txt}"

mkdir -p "$RUN_DIR"
: > "$LOG_PATH"
exec >>"$LOG_PATH" 2>&1

echo "[overnight] started: $(date --iso-8601=seconds)"
echo "[overnight] repo: $REPO_ROOT"
echo "[overnight] tag: $TAG"
echo "[overnight] rocm_base: $ROCM_BASE"
echo "[overnight] run_dir: $RUN_DIR"

SMOKE_OUT="$RUN_DIR/flux_smoke_256.png"
SAMPLE_OUT="$RUN_DIR/flux_sample_512.png"

sudo docker build --no-cache \
  --build-arg ROCM_BASE="$ROCM_BASE" \
  -t "$TAG" \
  "$REPO_ROOT/stable-diffusion"

echo "[overnight] build complete: $(date --iso-8601=seconds)"

cd "$REPO_ROOT"

CONTAINER_IMAGE="$TAG" \
OUT_PATH="$SMOKE_OUT" \
HEIGHT=256 \
WIDTH=256 \
STEPS=4 \
GUIDANCE=3.0 \
SEED=123 \
MODEL_CPU_OFFLOAD=1 \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 \
PROMPT='abstract iridescent glass petals floating over dark water, soft reflections, photoreal, high detail' \
stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh

CONTAINER_IMAGE="$TAG" \
OUT_PATH="$SAMPLE_OUT" \
HEIGHT=512 \
WIDTH=512 \
STEPS=8 \
GUIDANCE=3.0 \
SEED=321 \
MODEL_CPU_OFFLOAD=1 \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 \
PROMPT='luminous iridescent glass flowers floating over dark water at dawn, cinematic reflections, ultra detailed, elegant, surreal but photoreal' \
stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh

RUN_DIR_FOR_REPORT="$RUN_DIR" python - <<'PY' > "$REPORT_PATH"
from pathlib import Path
from PIL import Image
from datetime import datetime
import os

run_dir = Path(os.environ["RUN_DIR_FOR_REPORT"])
smoke = run_dir / "flux_smoke_256.png"
sample = run_dir / "flux_sample_512.png"

print(f"completed_at={datetime.now().isoformat()}")
for path in (smoke, sample):
    if not path.exists():
        print(f"missing={path}")
        continue
    img = Image.open(path)
    print(f"path={path}")
    print(f"size={img.size}")
    print(f"mode={img.mode}")
    print(f"bytes={path.stat().st_size}")
PY

echo "[overnight] report written: $REPORT_PATH"
echo "[overnight] completed: $(date --iso-8601=seconds)"
