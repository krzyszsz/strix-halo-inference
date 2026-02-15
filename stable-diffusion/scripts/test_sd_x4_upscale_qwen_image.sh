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

MODEL_ID="${MODEL_ID:-$MODEL_ROOT/sd-x4-upscaler}"
INPUT_PATH="${INPUT_PATH:-$REPO_ROOT/qwen-image/out/qwen_image_1024_attempt2.png}"
OUT_DIR="$REPO_ROOT/stable-diffusion/out"
OUT_PATH="${OUT_PATH:-$OUT_DIR/qwen_image_upscaled_sd_x4.png}"
PROMPT="${PROMPT:-cinematic portrait photo, natural skin texture, realistic lighting, fine facial detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-plastic skin, waxy skin, over-smoothed texture, low detail}"
INPUT_SIDE="${INPUT_SIDE:-512}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-7.5}"
NOISE_LEVEL="${NOISE_LEVEL:-20}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda}"
AOTRITON_EXPERIMENTAL="${AOTRITON_EXPERIMENTAL:-0}"
DISABLE_SDP="${DISABLE_SDP:-1}"

mkdir -p "$OUT_DIR"

if [ ! -f "$INPUT_PATH" ]; then
  echo "Input image not found: $INPUT_PATH" >&2
  exit 1
fi

TMP_PY="$(mktemp "${TMPDIR:-/tmp}/sd_x4_upscale.XXXXXX.py")"
trap 'rm -f "$TMP_PY"' EXIT

cat <<'PY' > "$TMP_PY"
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image


def resolve_dtype(name: str) -> torch.dtype:
    if name.lower() in {"float32", "fp32", "f32"}:
        return torch.float32
    if name.lower() in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float16


model_id = os.environ["MODEL_ID"]
input_path = Path(os.environ["INPUT_PATH"])
out_path = Path(os.environ["OUT_PATH"])
prompt = os.environ["PROMPT"]
negative_prompt = os.environ["NEGATIVE_PROMPT"]
input_side = int(os.environ["INPUT_SIDE"])
steps = int(os.environ["STEPS"])
guidance = float(os.environ["GUIDANCE"])
noise_level = int(os.environ["NOISE_LEVEL"])
dtype_name = os.environ.get("DTYPE", "float16")
dtype = resolve_dtype(dtype_name)
device = os.environ.get("DEVICE", "cuda")

img = Image.open(input_path).convert("RGB")
if img.size != (input_side, input_side):
    img = img.resize((input_side, input_side), Image.LANCZOS)

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
)
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()
pipe = pipe.to(device)

try:
    with torch.inference_mode():
        def cb(step: int, timestep: int, latents):
            # Ensure progress is visible in non-interactive logs (prevents watchdog "idle" false positives).
            print(f"[progress] step={step+1}/{steps} timestep={int(timestep)}", flush=True)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            num_inference_steps=steps,
            guidance_scale=guidance,
            noise_level=noise_level,
            callback=cb,
            callback_steps=1,
        ).images[0]
except Exception:
    if dtype != torch.float32:
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        pipe = pipe.to(device)
        with torch.inference_mode():
            def cb(step: int, timestep: int, latents):
                print(f"[progress] step={step+1}/{steps} timestep={int(timestep)}", flush=True)

            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                num_inference_steps=steps,
                guidance_scale=guidance,
                noise_level=noise_level,
                callback=cb,
                callback_steps=1,
            ).images[0]
    else:
        raise

out_path.parent.mkdir(parents=True, exist_ok=True)
out.save(out_path)
print(out_path, out.size, out_path.stat().st_size)
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
  -e INPUT_PATH="$INPUT_PATH" \
  -e OUT_PATH="$OUT_PATH" \
  -e PROMPT="$PROMPT" \
  -e NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
  -e INPUT_SIDE="$INPUT_SIDE" \
  -e STEPS="$STEPS" \
  -e GUIDANCE="$GUIDANCE" \
  -e NOISE_LEVEL="$NOISE_LEVEL" \
  -e DTYPE="$DTYPE" \
  -e DEVICE="$DEVICE" \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$AOTRITON_EXPERIMENTAL" \
  -e PYTORCH_SDP_DISABLE_FLASH_ATTENTION="$DISABLE_SDP" \
  -e PYTORCH_SDP_DISABLE_MEM_EFFICIENT="$DISABLE_SDP" \
  -v "$TMP_PY:/tmp/sd_x4_upscale.py:Z" \
  --entrypoint python \
  stable-diffusion-rocm:latest \
  -u /tmp/sd_x4_upscale.py &
pid="$!"

while kill -0 "$pid" >/dev/null 2>&1; do
  echo "Waiting for upscale to finish... (pid=$pid)"
  sleep 30
done
wait "$pid"

if [ -f "$OUT_PATH" ] && [ ! -w "$OUT_PATH" ]; then
  sudo chown "$(id -u):$(id -g)" "$OUT_PATH" || true
fi
