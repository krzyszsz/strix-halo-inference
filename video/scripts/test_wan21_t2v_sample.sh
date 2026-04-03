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

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/wan21-t2v-1.3b-diffusers}"
OUT_DIR="$REPO_ROOT/video/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/wan21_t2v_sample.mp4}"
PROMPT="${PROMPT:-Cinematic shot of a small robot barista serving coffee in a cozy cafe, shallow depth of field, realistic lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, distorted anatomy, jitter}"
WIDTH="${WIDTH:-672}"
HEIGHT="${HEIGHT:-384}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
STEPS="${STEPS:-8}"
GUIDANCE="${GUIDANCE:-5.0}"
SEED="${SEED:-1234}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-1}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-stable-diffusion-rocm:latest}"

mkdir -p "$OUT_DIR"

if [ ! -d "$MODEL_ID" ]; then
  echo "Model directory not found: $MODEL_ID" >&2
  exit 1
fi

TMP_PY="$(mktemp "${TMPDIR:-/tmp}/wan21_t2v.XXXXXX.py")"
trap 'rm -f "$TMP_PY"' EXIT

cat <<'PY' > "$TMP_PY"
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from PIL import Image


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


model_id = os.environ["MODEL_ID"]
out_path = Path(os.environ["OUT_PATH"])
prompt = os.environ["PROMPT"]
negative_prompt = os.environ["NEGATIVE_PROMPT"]
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
num_frames = int(os.environ["NUM_FRAMES"])
fps = int(os.environ["FPS"])
steps = int(os.environ["STEPS"])
guidance = float(os.environ["GUIDANCE"])
seed = int(os.environ["SEED"])
dtype = resolve_dtype(os.environ.get("DTYPE", "bfloat16"))
enable_cpu_offload = os.environ.get("ENABLE_CPU_OFFLOAD", "1") == "1"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()

if enable_cpu_offload:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

generator = torch.Generator("cpu").manual_seed(seed)

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=steps,
    guidance_scale=guidance,
    generator=generator,
)

frames = output.frames[0]
converted = []
for frame in frames:
    if isinstance(frame, Image.Image):
        arr = np.array(frame.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    converted.append(arr)

if not converted:
    raise RuntimeError("No frames generated")

h, w = converted[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

for frame in converted:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()

print(f"{out_path} frames={len(converted)} fps={fps} size={out_path.stat().st_size}")
PY

docker run --rm \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host \
  -v "$HF_ROOT:$HF_ROOT" \
  -v "$REPO_ROOT:$REPO_ROOT:Z" \
  -w "$REPO_ROOT" \
  -e MODEL_ID="$MODEL_ID" \
  -e OUT_PATH="$OUT_PATH" \
  -e PROMPT="$PROMPT" \
  -e NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
  -e WIDTH="$WIDTH" \
  -e HEIGHT="$HEIGHT" \
  -e NUM_FRAMES="$NUM_FRAMES" \
  -e FPS="$FPS" \
  -e STEPS="$STEPS" \
  -e GUIDANCE="$GUIDANCE" \
  -e SEED="$SEED" \
  -e DTYPE="$DTYPE" \
  -e ENABLE_CPU_OFFLOAD="$ENABLE_CPU_OFFLOAD" \
  -v "$TMP_PY:/tmp/wan21_t2v.py:Z" \
  --entrypoint python \
  "$CONTAINER_IMAGE" \
  -u /tmp/wan21_t2v.py

if [ -f "$OUT_PATH" ] && [ ! -w "$OUT_PATH" ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_PATH" || true
fi
