# Stable Diffusion (ROCm)

Diffusers-based ROCm container supporting SD3.5, SDXL, and Flux variants on Strix Halo with a single REST API.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

### BUILD

```bash
cd $REPO_ROOT/stable-diffusion
DOCKER_BUILDKIT=1 docker build -t stable-diffusion-rocm:latest .
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

Run the API (defaults to `$MODEL_ROOT/sd35-medium`):

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v $HF_ROOT:$HF_ROOT \
  -e HF_HOME=$HF_ROOT \
  -e MODEL_ID=$MODEL_ROOT/sd35-medium \
  -e DTYPE=float32 \
  stable-diffusion-rocm:latest
```

If you need to download a model, pass the read-only HF token at runtime:

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
  -e MODEL_ID=stabilityai/stable-diffusion-3.5-medium \
  -e DTYPE=float32 \
  stable-diffusion-rocm:latest
```

Download SDXL base (larger, modern baseline):

```bash
$REPO_ROOT/stable-diffusion/scripts/download_sdxl_base.sh
```

Download SD3.5 weights:

```bash
$REPO_ROOT/stable-diffusion/scripts/download_sd35_medium.sh
$REPO_ROOT/stable-diffusion/scripts/download_sd35_large.sh
```

Download a high-quality SDXL fine-tune (good instruction-following; 1024px-native):

```bash
$REPO_ROOT/stable-diffusion/scripts/download_playground_v25.sh
```

Download a Flux model (modern, higher quality baseline):

```bash
$REPO_ROOT/stable-diffusion/scripts/download_flux2_klein_4b.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_klein_9b.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_dev.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_dev_bnb4.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_dev_nvfp4.sh
```

Generate an image:

```bash
curl -s http://127.0.0.1:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$MODEL_ROOT/sd35-medium",
    "prompt": "A futuristic tram in the rain, neon reflections, ultra-detailed",
    "parameters": {
      "num_inference_steps": 40,
      "guidance_scale": 4.5,
      "height": 768,
      "width": 768,
      "seed": 123
    }
  }' | jq -r '.data[0].b64_json' | base64 -d > sd35.png
```

Recommended settings (model card / pipeline defaults):
- SD3.5 Medium: `num_inference_steps=40`, `guidance_scale=4.5`.
- SD3.5 Large: `num_inference_steps=28`, `guidance_scale=4.5` (on this host `512×512` is validated with fewer steps; `1024×1024` is not recommended).
- SDXL Base: `num_inference_steps=50`, `guidance_scale=5.0`.
- Playground v2.5: `num_inference_steps=20`, `guidance_scale=3.0`, `1024×1024`.
- Flux2-klein-4B: `num_inference_steps=4`, `guidance_scale=1.0`.
- FLUX.2 docs note minimum GPU memory of roughly `24GB` for `[dev]`, `16GB` for `[pro]`, `12GB` for `[schnell]` (availability/licensing and format support still apply).

Saved test outputs:
- `$REPO_ROOT/stable-diffusion/out/sd35_sample_best_retest.png`
- `$REPO_ROOT/stable-diffusion/out/sd35_large_512_best_attempt.png`
- `$REPO_ROOT/stable-diffusion/out/sdxl_base_best_retest.png`
- `$REPO_ROOT/stable-diffusion/out/playground_v25_1024_2026-02-12.png`
- `$REPO_ROOT/stable-diffusion/out/flux2_klein_best_retest.png`
- `$REPO_ROOT/stable-diffusion/out/flux2_klein_base_4b_512_2026-02-11.png`
- `$REPO_ROOT/stable-diffusion/out/flux2_klein_9b_512_t2i_reconfirm_2026-02-12.png`
- `$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_t2i_reconfirm_2026-02-12.png`
- `$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png`
- `$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png`

### SCRIPTS

- `scripts/run_server.sh`: start the container with specified model and dtype.
- `scripts/test_sd35_sample.sh`: start server, generate a small SD3.5 image, save output.
- `scripts/test_sdxl_base_sample.sh`: start server, generate a small SDXL image, save output.
- `scripts/test_flux2_klein_sample.sh`: start server, generate a Flux2 image, save output.
- `scripts/test_playground_v25_sample.sh`: start server, generate a Playground v2.5 image (`1024×1024`), save output.
- `scripts/download_sd35_medium.sh`: download SD3.5 medium weights.
- `scripts/download_sd35_large.sh`: download SD3.5 large weights.
- `scripts/download_sdxl_base.sh`: download SDXL base weights.
- `scripts/download_playground_v25.sh`: download Playground v2.5 weights (SDXL fine-tune, 1024px-native).
- `scripts/download_flux2_klein_4b.sh`: download Flux2 Klein 4B weights.
- `scripts/download_flux2_klein_9b.sh`: download Flux2 Klein 9B weights (gated access required).
- `scripts/download_flux2_dev.sh`: download FLUX.2-dev weights (gated access required).
- `scripts/download_flux2_dev_bnb4.sh`: download `diffusers/FLUX.2-dev-bnb-4bit` (practical `dev`-family variant on this host).
- `scripts/download_flux2_dev_nvfp4.sh`: download FLUX.2-dev-NVFP4 checkpoint files.

Notes:
- For slow networks, prefer local paths under `$MODEL_ROOT`.
- Default dtype is FP16 (`DTYPE=float16`). On RDNA3.5/ROCm, FP16 can fail with type mismatch; set `DTYPE=float32` or let the API auto-fallback to FP32 when it detects the mismatch.
- Diffusers is installed from Git in this image to support newer pipelines like Flux2; rebuild the image after changing dependencies.
- The test scripts use smaller resolutions (e.g., 512–768) and fewer steps to keep runtime reasonable on Strix Halo. Increase to 1024+ and the recommended steps above for best quality.
- SD3.5‑large is slower and less predictable than SD3.5‑medium on this host; keep steps modest and prefer SD3.5‑medium / SDXL / FLUX.2 for reliability.
- `FLUX.2-klein-base-4B` was tested successfully on this host under the `75g` policy (`reports/publish/flux2_klein_base_4b_512.log`).
- `FLUX.2-klein-9B` and `FLUX.2-dev` are gated on HF. With authenticated access on this machine:
  - `FLUX.2-klein-9B` was validated at `512x512` (`STEPS=4`, bf16, CPU offload) under the `75g` policy.
  - Full `FLUX.2-dev` is currently not stable on this host (remote text-encoder 503s, local path triggers host OOM even at tiny sizes).
  - Practical `dev`-family option: `diffusers/FLUX.2-dev-bnb-4bit` (saved under `$MODEL_ROOT/flux2-dev-bnb4`) validated at `512x512` for text-to-image, image-to-image, and multi-image composition.
- `FLUX.2-dev-NVFP4` downloads, but this container currently expects a full diffusers directory (it does not load checkpoint-only repos directly).
