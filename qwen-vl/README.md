# Qwen-VL (ROCm)

Local vision-language container for Strix Halo. It accepts base64 images + prompts and returns a textual description or critique.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

### BUILD

```bash
cd $REPO_ROOT/qwen-vl
DOCKER_BUILDKIT=1 docker build -f Dockerfile.qwen-vl.rocm -t qwen-vl-rocm:latest .
```

### HOST SETUP (Fedora 43 / Strix Halo)

This follows the same kernel/UMA tuning as the other containers.

### USAGE & EXAMPLES

Run the API (defaults to `Qwen/Qwen2.5-VL-7B-Instruct`):

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v $HF_ROOT:$HF_ROOT \
  -e HF_HOME=$HF_ROOT \
  -e MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
  -e DTYPE=float16 \
  -e PORT=8005 \
  qwen-vl-rocm:latest
```

Send a base64 image for description:

```bash
FORCE_CPU=1 \
INPUT_IMAGE=$REPO_ROOT/qwen-vl/input/qwen_image_full_256.png \
OUT_PATH=$REPO_ROOT/qwen-vl/out/qwen_vl_describe_75g_retest2.txt \
MAX_NEW_TOKENS=256 \
CURL_MAX_TIME=900 \
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/qwen-vl/scripts/test_qwen_vl_7b.sh
```

Saved test output:
- `$REPO_ROOT/qwen-vl/out/qwen_vl_describe_75g_retest2.txt` (CPU fallback run on a 256×256 test image)

### SCRIPTS

- `scripts/download_qwen_vl_7b.sh`: download the model weights to `$MODEL_ROOT/qwen2.5-vl-7b-instruct`.
- `scripts/test_qwen_vl_7b.sh`: start the server, send a test image, save response.

### NOTES

- The Qwen2.5‑VL docs recommend constraining image pixels with `min_pixels=256*28*28` and `max_pixels=1280*28*28`. This container exposes them as `MIN_PIXELS` and `MAX_PIXELS` environment variables.
- ROCm/GPU runs caused hardware exceptions on this host; CPU fallback (`FORCE_CPU=1`) with a **small image** and lower token budget succeeded.
