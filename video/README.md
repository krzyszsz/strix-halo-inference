# Video Generation (ROCm)

Short text-to-video validation workflow for Strix Halo using `Wan2.1-T2V-1.3B` via Diffusers.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

This workflow reuses the `stable-diffusion-rocm:latest` container from `stable-diffusion/`.

## Download

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/video/scripts/download_wan21_t2v_1_3b.sh
```

Default model path:
- `$MODEL_ROOT/wan21-t2v-1.3b-diffusers`

## Test Run

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/video/scripts/test_wan21_t2v_sample.sh
```

Default validated profile:
- `WIDTH=672`
- `HEIGHT=384`
- `NUM_FRAMES=17`
- `FPS=8`
- `STEPS=8`
- `GUIDANCE=5.0`
- `DTYPE=bfloat16`
- `ENABLE_CPU_OFFLOAD=1`
- container memory cap: `MEM_LIMIT=75g` / `MEMORY_SWAP=75g`

Outputs:
- `video/out/wan21_t2v_sample.mp4`
- `video/out/wan21_t2v_sample_frame0.png` (frame extracted with `ffmpeg`)

Evidence logs:
- `reports/publish/wan21_t2v.log`
