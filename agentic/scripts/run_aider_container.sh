#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AGENTIC_DIR="$REPO_ROOT/agentic"

docker() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

PORT="${PORT:-8004}"
MODEL="${MODEL:-openai/local-gguf}"
API_KEY="${OPENAI_API_KEY:-dummy}"
LOG_PATH="${LOG_PATH:-$AGENTIC_DIR/out/aider_container.log}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$(dirname "$LOG_PATH")"

docker run --rm -i --network=host \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  -v "$AGENTIC_DIR:/workspace:Z" \
  -w /workspace/dotnet-demo \
  python:3.12-slim-bookworm bash -lc \
  "export DEBIAN_FRONTEND=noninteractive \
   && apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 update \
   && apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 install -y --no-install-recommends git ca-certificates \
   && rm -rf /var/lib/apt/lists/* \
   && pip install --no-cache-dir aider-chat \
   && git init \
   && aider --model ${MODEL} \
        --openai-api-base http://127.0.0.1:${PORT}/v1 \
        --openai-api-key ${API_KEY} \
        --message-file /workspace/prompts/dotnet_minimal_api.txt \
        --no-show-model-warnings \
        --no-show-release-notes \
        --no-browser \
        --yes-always" | tee "$LOG_PATH"
