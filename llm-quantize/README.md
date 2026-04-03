# Local Quantization (Demo)

This folder demonstrates a reproducible, containerized GGUF quantization workflow:

- Download a small **FP16** GGUF model to the host volume (`$MODEL_ROOT`)
- Quantize **FP16 -> Q4_K_M** inside a Docker container (`llama-quantize`)
- Verify the quantized model still answers a text prompt (`llama-cli`)

The host stays clean: all tooling lives in the container. Model files and outputs are written to the host via bind mounts.

Build:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"

DOCKER_BUILDKIT=1 docker build -t llama-cpp-tools:latest "$REPO_ROOT/llm-quantize"
```

Run the end-to-end demo (with repo-wide memory cleanup wrapper):

```bash
scripts/run_memsafe.sh bash llm-quantize/scripts/quantize_qwen25_05b_fp16_to_q4km.sh
```

Evidence outputs land under:
- `reports/quantize/`

