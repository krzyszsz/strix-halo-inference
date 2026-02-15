# Qwen-Image-Edit (ROCm)

Image-to-image editing container for Strix Halo. It wraps Qwen Image Edit pipelines behind a REST API that accepts base64 inputs.
On this host:
- `Qwen/Qwen-Image-Edit` is used as the stable single-image edit path.
- `Qwen/Qwen-Image-Edit-2509` and `Qwen/Qwen-Image-Edit-2511` (Plus variants, `1-3` images) are now stable with the tuned profile:
  - `DTYPE=bfloat16`
  - `ENABLE_SEQUENTIAL_CPU_OFFLOAD=1`
  - `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  - `TRUE_CFG_SCALE=1.0`
  - `INCLUDE_NEGATIVE_PROMPT=0`
  - `MAX_SEQUENCE_LENGTH=128`
  - memory profile used in successful runs: `MEM_LIMIT=75g`, `MEMORY_SWAP=140g`, `MEM_RESERVATION=67g`

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

### BUILD

```bash
cd $REPO_ROOT/qwen-image-edit
DOCKER_BUILDKIT=1 docker build -t qwen-image-edit-rocm:latest .
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

Run the API (defaults to Qwen/Qwen-Image-Edit):

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v $HF_ROOT:$HF_ROOT \
  -e HF_HOME=$HF_ROOT \
  -e MODEL_ID=Qwen/Qwen-Image-Edit \
  -e DTYPE=bfloat16 \
  qwen-image-edit-rocm:latest
```

If the model is missing, pass the read-only HF token:

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
  -e MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
  -e DTYPE=bfloat16 \
  qwen-image-edit-rocm:latest
```

Prepare a base64 image payload:

```bash
IMAGE_B64=$(base64 -w 0 ./input.png)
```

Edit an image via REST:

```bash
curl -s http://127.0.0.1:8002/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-Edit",
    "prompt": "Replace the sky with a dramatic sunset",
    "image": "'"$IMAGE_B64"'",
    "parameters": {
      "num_inference_steps": 50,
      "true_cfg_scale": 4.0,
      "negative_prompt": " ",
      "strength": 0.7
    }
  }' | jq -r '.data[0].b64_json' | base64 -d > edited.png
```

Multi-image edit via REST (Plus pipeline only):

```bash
IMAGE_A_B64=$(base64 -w 0 ./input_a.png)
IMAGE_B_B64=$(base64 -w 0 ./input_b.png)

curl -s http://127.0.0.1:8002/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-Edit-2509",
    "prompt": "Combine both source images into one coherent scene.",
    "images_b64": ["'"$IMAGE_A_B64"'", "'"$IMAGE_B_B64"'"],
    "parameters": {
      "num_inference_steps": 8,
      "true_cfg_scale": 1.0,
      "max_sequence_length": 128,
      "height": 512,
      "width": 512
    }
  }'
```

Repeatable script for tuned Plus profile (single or multi):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=single MODEL_ID=$MODEL_ROOT/qwen-image-edit-2511 HEIGHT=512 WIDTH=512 STEPS=4 \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2511_single_512_seqoffload_bf16_75g_swap140_test.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh

$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=multi MODEL_ID=$MODEL_ROOT/qwen-image-edit-2509 HEIGHT=512 WIDTH=512 STEPS=8 \
      INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_single_compat_2026-02-11.png \
      INPUT_IMAGE_B=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png \
      PROMPT='Create one coherent scene by placing the human from image A into image B. Keep the person identity and face natural, match lighting and perspective, keep both subjects visible.' \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2509_multi_512_human_insert_steps8_75g_swap140.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh
```

Saved test outputs:
- Base model:
  - `qwen-image-edit/out/qwen_image_edit_single_compat_2026-02-11.png`
- Plus models (`2509/2511`):
  - `qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png`
  - `qwen-image-edit/out/qwen_image_edit_2509_multi_512_human_insert_steps8_75g_swap140.png`
  - `qwen-image-edit/out/qwen_image_edit_2511_single_512_seqoffload_bf16_75g_swap140_test.png`
  - `qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps12_cfg2_seed3456_75g_swap140_2026-02-13.png`

### SCRIPTS

- `scripts/run_server.sh`: start the container with specified model and dtype.
- `scripts/test_qwen_image_edit.sh`: start server, edit a full-size image, save output.
- `scripts/download_qwen_image_edit.sh`: download full Qwen-Image-Edit weights.
- `scripts/download_qwen_image_edit_2511.sh`: download Qwen-Image-Edit-2511 (latest Plus) weights.
- `scripts/download_qwen_image_edit_2509.sh`: download Qwen-Image-Edit-2509 (Plus/multi-image) weights.
- `scripts/test_qwen_image_edit_multi_input.sh`: send two input images to Plus pipeline and save output when successful.
- `scripts/test_qwen_image_edit_plus_stable.sh`: wrapper that applies the tuned Plus settings and dispatches to single/multi test scripts.

Notes:
- For slow networks, prefer local paths under `$MODEL_ROOT`.
- Default dtype is **bfloat16** on this host. FP16 can trigger type-mismatch errors, while FP32 can OOM for full-size runs.
- If a model ignores certain parameters (for example, `strength`), the API filters unsupported keys automatically.
- Models with suffix `-2511` or `-2509` use the newer “Plus” pipeline; the container selects this automatically.
- Recommended baseline parameters:
  - Base model (`Qwen-Image-Edit`): `num_inference_steps=50`, `true_cfg_scale=4.0`, `negative_prompt=" "`.
  - Plus models (`2509/2511`) on this host: `num_inference_steps=4..8`, `true_cfg_scale=1.0`, no negative prompt, `max_sequence_length=128`, `bfloat16`, and sequential CPU offload.
- To avoid long silent waits during startup diagnostics, use:
  - `HEALTH_RETRIES` and `HEALTH_SLEEP` in test scripts (`test_qwen_image_edit.sh`, `test_qwen_image_edit_multi_input.sh`).
