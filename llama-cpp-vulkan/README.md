# llama.cpp Vulkan Server

Vulkan-enabled llama.cpp server for running GGUF models on Strix Halo without ROCm dependencies.

Set once:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
```

### BUILD

```bash
cd $REPO_ROOT/llama-cpp-vulkan
DOCKER_BUILDKIT=1 docker build -t llama-cpp-vulkan:latest .
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

Download a small GGUF model first (tiny/quantized is best for initial validation):

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --include "*Q4_K_M*.gguf" \
  --local-dir $MODEL_ROOT/qwen2.5-1.5b-instruct-gguf \
  --local-dir-use-symlinks False
```

Run the server:

```bash
docker run --rm -it \
  --device=/dev/dri \
  --security-opt label=disable \
  --ipc=host --network=host \
  -v $HF_ROOT:$HF_ROOT \
  -e MODEL=$MODEL_ROOT/qwen2.5-1.5b-instruct-gguf/<your-model>.gguf \
  llama-cpp-vulkan:latest
```

Download a larger model (Qwen3-Next 80B, Q5_K_M):

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen3_next_80b.sh
```

Download dense/fallback models:

```bash
# (Optional) dense fallbacks exist, but this repo's publish-day validation focuses on the
# Next MoE models + Qwen2.5-Coder + GPT-OSS (see root README.md for the evidence table).
```

These are useful if the MoE “Next” models are unstable on your build or if you want a smaller dense fallback with simpler memory behavior.

Run Qwen3-Next 80B:

```bash
MODEL=$MODEL_ROOT/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf \
CTX_SIZE=2048 GPU_LAYERS=999 THREADS=8 \
  $REPO_ROOT/llama-cpp-vulkan/scripts/run_server.sh
```

Download a coder model (Qwen3-Coder-Next, Q5_K_M):

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen3_coder_next.sh
```

Run Qwen3-Coder-Next (Q5_K_M, sharded):

```bash
PORT=8004 \
MODEL=$MODEL_ROOT/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf \
CTX_SIZE=2048 GPU_LAYERS=999 THREADS=8 \
  $REPO_ROOT/llama-cpp-vulkan/scripts/run_server.sh
```

Fallback coder (Qwen2.5-Coder 32B, Q4_K_M):

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen25_coder_32b.sh
```

Run Qwen2.5-Coder 32B:

```bash
PORT=8004 \
MODEL=$MODEL_ROOT/qwen2.5-coder-32b-instruct-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf \
CTX_SIZE=2048 GPU_LAYERS=999 THREADS=8 \
  $REPO_ROOT/llama-cpp-vulkan/scripts/run_server.sh
```

OpenAI-style chat example:

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-gguf",
    "messages": [{"role": "user", "content": "Write a 1-sentence summary of Vulkan."}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

Saved test outputs (publish-day rerun artifacts):
- `$REPO_ROOT/llama-cpp-vulkan/out/qwen3_next_80b_q5_ctx_196608_75g_retest2.json`
- `$REPO_ROOT/llama-cpp-vulkan/out/qwen3_coder_next_q5_ctx_196608_75g_retest2.json`
- `$REPO_ROOT/llama-cpp-vulkan/out/qwen25_coder_32b_q4_ctx_131072_75g_retest2.json`
- `$REPO_ROOT/llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_ctx_131072_75g_retest2.json`

### Recommended Sampling + Context Notes

Qwen3-Next best-practice sampling (from the model card):

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0`
- recommended output length: `16384` tokens

Context lengths:

- **Qwen3-Next-80B-A3B**: 262,144 tokens natively (can be extended with YaRN).
- **Qwen2.5-Coder-32B-Instruct**: 131,072 tokens (model card); this repo validated `ctx=131072` on the GGUF Q4_K_M build.

In this repo, tests start at `CTX_SIZE=2048` for stability, then increase gradually.
If the server fails to start at high context, drop back to ~32k as suggested in the model card and/or reduce KV cache precision.
On this host, a 32k context smoke test succeeded with Qwen3‑Next Q5 (`CTX_SIZE=32768`).
For Qwen2.5-Coder GGUF, no explicit sampling defaults are listed in the model card; this repo uses the Qwen3 best-practice sampler as a conservative baseline.

### SCRIPTS

- `scripts/run_server.sh`: start the container with a specified GGUF model.
- `scripts/download_qwen3_next_80b.sh`: download Qwen3-Next 80B GGUF (Q5_K_M).
- `scripts/download_qwen3_coder_next.sh`: download Qwen3-Coder-Next GGUF (Q5_K_M).
- `scripts/download_qwen3_32b_q6.sh`: download Qwen3-32B GGUF (Q6_K).
- `scripts/download_qwen3_coder_30b_q8.sh`: download Qwen3-Coder-30B GGUF (Q8_0).
- `scripts/download_qwen25_coder_32b.sh`: download Qwen2.5-Coder 32B GGUF (Q4_K_M).
- `scripts/test_qwen3_next_80b.sh`: start server, query chat, save output JSON.
- `scripts/test_qwen3_coder_next_q5.sh`: start server, query coder prompt, save output JSON.
- `scripts/test_qwen3_32b_q6.sh`: start server, query general prompt, save output JSON.
- `scripts/test_qwen3_coder_30b_q8.sh`: start server, query coder prompt, save output JSON.
- `scripts/test_qwen25_coder_32b.sh`: start server, query coder prompt, save output JSON.
- `scripts/write_continue_config.sh`: emit a Continue config example.

Notes:
- Start with a small model (Qwen-1.5B or Llama-3-8B) before trying Gemma-27B.
- If Vulkan ICDs are not detected, bind-mount the host ICDs into the container:
  `-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro`.
- For large models, tune `CTX_SIZE`, `GPU_LAYERS`, and `THREADS` if you hit memory or speed issues.

### VS CODE (CONTINUE) SETUP

1) Install the Continue extension in VS Code.
2) Generate a config example:

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/write_continue_config.sh
```

3) Copy the generated file to your Continue config path (for example `~/.continue/config.yaml`) and adjust the ports if needed.

Continue supports OpenAI-compatible providers via `apiBase`, so you can point it at the local llama.cpp server (e.g., `http://localhost:8003/v1`).
