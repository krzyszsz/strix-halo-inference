#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v aider >/dev/null 2>&1; then
  echo "aider not found. Run $REPO_ROOT/agentic/scripts/install_aider.sh or use run_aider_container.sh" >&2
  exit 1
fi

PORT="${PORT:-8004}"
MODEL="${MODEL:-openai/local-gguf}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/agentic/prompts/dotnet_minimal_api.txt}"
WORKDIR="${WORKDIR:-$REPO_ROOT/agentic/dotnet-demo}"
API_KEY="${OPENAI_API_KEY:-dummy}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/agentic/out/aider_dotnet_demo.log}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
mkdir -p "$(dirname "$LOG_PATH")"

if [ ! -d .git ]; then
  git init
fi

aider \
  --model "$MODEL" \
  --openai-api-base "http://127.0.0.1:${PORT}/v1" \
  --openai-api-key "$API_KEY" \
  --message-file "$PROMPT_FILE" \
  --no-show-model-warnings \
  --no-show-release-notes \
  --no-browser \
  --yes-always | tee "$LOG_PATH"
