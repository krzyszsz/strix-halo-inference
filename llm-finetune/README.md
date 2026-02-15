# Fine-Tuning (Tiny LoRA Demo)

This folder demonstrates a **small**, **reproducible**, containerized fine-tuning workflow:

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0)
- Method: LoRA SFT (few steps, short sequences)
- Hardware mode: CPU (keeps this portable and avoids ROCm/driver variability)

All training runs inside a Docker container; model downloads and adapter outputs are stored on the host (`$MODEL_ROOT`) via bind mounts.

Build:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"

DOCKER_BUILDKIT=1 docker build -t llm-finetune:latest "$REPO_ROOT/llm-finetune"
```

Run the end-to-end demo (with repo-wide memory cleanup wrapper):

```bash
scripts/run_memsafe.sh bash llm-finetune/scripts/finetune_smollm2_135m_lora_demo.sh
```

Evidence outputs land under:
- `reports/finetune/`

