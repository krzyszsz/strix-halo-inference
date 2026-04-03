#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUT_PATH="${1:-$REPO_ROOT/llama-cpp-vulkan/out/continue_config.yaml}"
mkdir -p "$(dirname "$OUT_PATH")"

cat <<'YAML' > "$OUT_PATH"
name: StrixHalo Local Models
version: 0.0.1
schema: v1

models:
  - name: Qwen3-Next-80B-A3B (llama.cpp)
    provider: openai
    model: local-gguf
    apiBase: http://localhost:8003/v1
    apiKey: unused
    roles:
      - chat
      - edit
      - apply

  - name: Qwen3-Coder-Next (llama.cpp)
    provider: openai
    model: local-gguf
    apiBase: http://localhost:8004/v1
    apiKey: unused
    roles:
      - autocomplete
YAML

echo "Wrote Continue config to $OUT_PATH"
