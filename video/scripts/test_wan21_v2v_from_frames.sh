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
OUT_DIR="${OUT_DIR:-$REPO_ROOT/video/out}"
OUT_PATH="${OUT_PATH:-$OUT_DIR/wan21_v2v_from_frames.mp4}"
OUT_FRAME0="${OUT_FRAME0:-$OUT_DIR/wan21_v2v_from_frames_frame0.png}"
OUT_INPUT_PREVIEW="${OUT_INPUT_PREVIEW:-$OUT_DIR/wan21_v2v_from_frames_input_preview.png}"
INPUT_IMAGE_A="${INPUT_IMAGE_A:-$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_a_512_seed1234.png}"
INPUT_IMAGE_B="${INPUT_IMAGE_B:-$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png}"
MODE="${MODE:-two}" # single|two
PROMPT="${PROMPT:-A smooth cinematic shot in a cozy coffee shop: two people stand naturally together while the camera performs a gentle handheld push-in, realistic lighting, detailed faces, stable composition.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-flicker, jitter, blur, low quality, distortions, extra limbs, warped faces}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
NUM_FRAMES="${NUM_FRAMES:-17}" # keep 4N+1
FPS="${FPS:-8}"
STEPS="${STEPS:-8}"
GUIDANCE="${GUIDANCE:-5.0}"
STRENGTH="${STRENGTH:-0.7}"
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
if [ ! -f "$INPUT_IMAGE_A" ]; then
  echo "Input image A not found: $INPUT_IMAGE_A" >&2
  exit 1
fi
if [ "$MODE" = "two" ] && [ ! -f "$INPUT_IMAGE_B" ]; then
  echo "Input image B not found: $INPUT_IMAGE_B" >&2
  exit 1
fi

TMP_PY="$(mktemp "${TMPDIR:-/tmp}/wan21_v2v_from_frames.XXXXXX.py")"
trap 'rm -f "$TMP_PY"' EXIT

cat <<'PY' > "$TMP_PY"
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import diffusers.pipelines.wan.pipeline_wan_video2video as wan_v2v_mod
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVideoToVideoPipeline


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


class _FallbackFtfy:
    @staticmethod
    def fix_text(text):
        return text


try:
    import ftfy as _ftfy  # type: ignore
except Exception:
    _ftfy = _FallbackFtfy()

# Diffusers main can reference ftfy in this module without importing it.
# Inject it explicitly so prompt cleaning doesn't raise NameError.
wan_v2v_mod.ftfy = _ftfy


def pil_rgb(path: str, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


model_id = os.environ["MODEL_ID"]
out_path = Path(os.environ["OUT_PATH"])
out_frame0 = Path(os.environ["OUT_FRAME0"])
out_input_preview = Path(os.environ["OUT_INPUT_PREVIEW"])
input_a = os.environ["INPUT_IMAGE_A"]
input_b = os.environ.get("INPUT_IMAGE_B", "")
mode = os.environ.get("MODE", "two").lower()
prompt = os.environ["PROMPT"]
negative_prompt = os.environ["NEGATIVE_PROMPT"]
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
num_frames = int(os.environ["NUM_FRAMES"])
fps = int(os.environ["FPS"])
steps = int(os.environ["STEPS"])
guidance = float(os.environ["GUIDANCE"])
strength = float(os.environ["STRENGTH"])
seed = int(os.environ["SEED"])
dtype = resolve_dtype(os.environ.get("DTYPE", "bfloat16"))
enable_cpu_offload = os.environ.get("ENABLE_CPU_OFFLOAD", "1") == "1"

if num_frames < 5 or (num_frames - 1) % 4 != 0:
    raise ValueError("NUM_FRAMES should be 4N+1 and at least 5")

image_a = pil_rgb(input_a, width, height)
image_b = pil_rgb(input_b, width, height) if (mode == "two" and input_b) else image_a

if mode == "single":
    input_video = [image_a.copy() for _ in range(num_frames)]
elif mode == "two":
    split = num_frames // 2
    input_video = [image_a.copy() for _ in range(split)] + [image_b.copy() for _ in range(num_frames - split)]
else:
    raise ValueError(f"Unsupported MODE='{mode}' (expected single|two)")

preview = np.hstack([np.array(image_a), np.array(image_b)])
out_input_preview.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(preview).save(out_input_preview)

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

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
    video=input_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_inference_steps=steps,
    guidance_scale=guidance,
    strength=strength,
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

out_path.parent.mkdir(parents=True, exist_ok=True)
out_frame0.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(converted[0]).save(out_frame0)

h, w = converted[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
for frame in converted:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()

print(f"{out_path} frames={len(converted)} fps={fps} size={out_path.stat().st_size}")
print(f"{out_frame0} size={out_frame0.stat().st_size}")
print(f"{out_input_preview} size={out_input_preview.stat().st_size}")
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
  -e OUT_FRAME0="$OUT_FRAME0" \
  -e OUT_INPUT_PREVIEW="$OUT_INPUT_PREVIEW" \
  -e INPUT_IMAGE_A="$INPUT_IMAGE_A" \
  -e INPUT_IMAGE_B="$INPUT_IMAGE_B" \
  -e MODE="$MODE" \
  -e PROMPT="$PROMPT" \
  -e NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
  -e WIDTH="$WIDTH" \
  -e HEIGHT="$HEIGHT" \
  -e NUM_FRAMES="$NUM_FRAMES" \
  -e FPS="$FPS" \
  -e STEPS="$STEPS" \
  -e GUIDANCE="$GUIDANCE" \
  -e STRENGTH="$STRENGTH" \
  -e SEED="$SEED" \
  -e DTYPE="$DTYPE" \
  -e ENABLE_CPU_OFFLOAD="$ENABLE_CPU_OFFLOAD" \
  -v "$TMP_PY:/tmp/wan21_v2v_from_frames.py:Z" \
  --entrypoint python \
  "$CONTAINER_IMAGE" \
  -u /tmp/wan21_v2v_from_frames.py

for f in "$OUT_PATH" "$OUT_FRAME0" "$OUT_INPUT_PREVIEW"; do
  if [ -f "$f" ] && [ ! -w "$f" ]; then
    sudo chown "$(id -u):$(id -g)" "$f" || true
  fi
done
