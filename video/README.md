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

## Frame-Conditioned V2V (one/two images)

The same `Wan2.1-T2V-1.3B` checkpoint can be used with Diffusers `WanVideoToVideoPipeline`.
In this repo, I build a short conditioning clip from one or two existing still images and run V2V.
Reference: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan#video-to-video-generation>

Script:
- `video/scripts/test_wan21_v2v_from_frames.sh`

Single-image conditioning (repeat one frame):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=single MODEL_ID=$MODEL_ROOT/wan21-t2v-1.3b-diffusers \
      INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png \
      PROMPT='Keep the same person and coffee shop composition; add gentle camera drift and natural blinking, realistic detail, stable face, smooth motion.' \
      WIDTH=512 HEIGHT=512 NUM_FRAMES=17 FPS=8 STEPS=8 GUIDANCE=5.0 STRENGTH=0.45 \
      OUT_PATH=$REPO_ROOT/video/out/wan21_v2v_singleframe_sample_2026-03-08.mp4 \
      OUT_FRAME0=$REPO_ROOT/video/out/wan21_v2v_singleframe_sample_frame0_2026-03-08.png \
      OUT_INPUT_PREVIEW=$REPO_ROOT/video/out/wan21_v2v_singleframe_input_preview_2026-03-08.png \
  bash $REPO_ROOT/video/scripts/test_wan21_v2v_from_frames.sh
```

Two-image conditioning (first half from A, second half from B):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=two MODEL_ID=$MODEL_ROOT/wan21-t2v-1.3b-diffusers \
      INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png \
      INPUT_IMAGE_B=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.png \
      PROMPT='Keep the same person and cafe scene; add slight cinematic camera sway and natural micro-movements while preserving identity and composition.' \
      WIDTH=512 HEIGHT=512 NUM_FRAMES=17 FPS=8 STEPS=8 GUIDANCE=5.0 STRENGTH=0.45 \
      OUT_PATH=$REPO_ROOT/video/out/wan21_v2v_twoframe_tuned_sample_2026-03-08.mp4 \
      OUT_FRAME0=$REPO_ROOT/video/out/wan21_v2v_twoframe_tuned_sample_frame0_2026-03-08.png \
      OUT_INPUT_PREVIEW=$REPO_ROOT/video/out/wan21_v2v_twoframe_tuned_input_preview_2026-03-08.png \
  bash $REPO_ROOT/video/scripts/test_wan21_v2v_from_frames.sh
```

Notes:
- Single-frame conditioning is visibly more stable with this `1.3B` checkpoint.
- Two-frame conditioning runs end-to-end but may show temporal/composition artifacts.
- Direct Wan I2V/FLF2V checkpoints are available upstream in larger variants; this repo test keeps the `1.3B` path already present locally.

Evidence logs:
- `reports/publish/wan21_v2v_singleframe_2026-03-08.log`
- `reports/publish/wan21_v2v_twoframe_tuned_2026-03-08.log`
