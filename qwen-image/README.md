# Qwen-Image (ROCm)

Text-to-image container built for Strix Halo using ROCm + diffusers. It exposes a small REST API for reproducible generation tests.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

### BUILD

```bash
cd $REPO_ROOT/qwen-image
DOCKER_BUILDKIT=1 docker build -f Dockerfile.qwen-image.rocm -t qwen-image-rocm:latest .
```

### HOST SETUP (Fedora 43 / Strix Halo)

Recommended kernel parameters for 96GB unified memory (conservative baseline):

```bash
sudo grubby --update-kernel=ALL --args="iommu=pt amdgpu.gttsize=73728 ttm.pages_limit=18874368 ttm.page_pool_size=18874368"
# Optional if you need more UMA headroom:
# sudo grubby --update-kernel=ALL --args="amdttm.pages_limit=18874368 amdttm.page_pool_size=18874368"
```

Notes:
- Avoid the reported 6.18.3-200 regression for Strix Halo; prefer a known-good adjacent 6.18.x build.
- Verify args after reboot with: `cat /proc/cmdline`.

### USAGE & EXAMPLES

Run the API (defaults to Qwen/Qwen-Image):

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v $HF_ROOT:$HF_ROOT \
  -e HF_HOME=$HF_ROOT \
  -e MODEL_ID=Qwen/Qwen-Image \
  -e DTYPE=bfloat16 \
  qwen-image-rocm:latest
```

Latest upstream variant test in this repo:
- `Qwen/Qwen-Image-2512` (downloaded and tested at `512x512`, `steps=20`)
- Evidence:
  - `reports/publish/qwen_image_2512_512.log`
  - `qwen-image/out/qwen_image_2512_512_2026-02-11.png`

If the model is not already in `$MODEL_ROOT`, pass the read-only HF token at runtime:

```bash
export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v $HF_ROOT:$HF_ROOT \
  -e HF_HOME=$HF_ROOT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  -e MODEL_ID=Qwen/Qwen-Image \
  -e DTYPE=bfloat16 \
  qwen-image-rocm:latest
```

Generate an image via REST (JSON parameters accepted):

```bash
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "A cinematic robot barista pouring latte art, warm lighting",
    "parameters": {
      "num_inference_steps": 50,
      "true_cfg_scale": 4.0,
      "negative_prompt": " ",
      "height": 1024,
      "width": 1024,
      "seed": 42
    }
  }' | jq -r '.data[0].b64_json' | base64 -d > qwen_image.png
```

Saved test outputs:
- `$REPO_ROOT/qwen-image/out/qwen_image_2512_512_2026-02-11.png`
- `$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png`
- `$REPO_ROOT/qwen-image/out/qwen_image_1024_75g_retest2.png`

### SCRIPTS

- `scripts/run_server.sh`: start the container with specified model and dtype.
- `scripts/test_qwen_image.sh`: start server, generate a full-model image, save output.
- `scripts/download_qwen_image.sh`: download full Qwen-Image weights.
- `scripts/download_qwen_image_2512.sh`: download latest Qwen-Image-2512 weights.

Notes:
- For slow networks, prefer local paths: set `model` to `$MODEL_ROOT/<model_dir>`.
- Default dtype is FP16 (`DTYPE=float16`). On this Strix Halo host, **FP16 produced black images**; `DTYPE=bfloat16` gave correct output, and `DTYPE=float32` hit OOM. Use bfloat16 for full-model runs.
- Recommended baseline parameters: `num_inference_steps=50`, `true_cfg_scale=4.0`, and `height/width=1024`. The test script defaults to 512×512 and 30 steps for runtime stability on this host; override `WIDTH/HEIGHT` and `STEPS` for full-size renders.
- On this machine, 1024×1024 at 50 steps can stall; use the 512×512/30-step preset as a stable baseline.
- `scripts/test_qwen_image.sh` supports `HEALTH_RETRIES` and `HEALTH_SLEEP` for startup diagnostics.
