#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command ...>" >&2
  echo "Example: $0 bash llama-cpp-vulkan/scripts/test_qwen3_next_80b.sh" >&2
  echo "Defaults: MEM_LIMIT=75g MEMORY_SWAP=75g MEM_RESERVATION=67g OOM_SCORE_ADJ=500" >&2
  exit 1
fi

export MEM_LIMIT="${MEM_LIMIT:-75g}"
export MEMORY_SWAP="${MEMORY_SWAP:-75g}"
export MEM_RESERVATION="${MEM_RESERVATION:-67g}"
export OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
export HF_ROOT
export MODEL_ROOT

RUN_WITH_WATCHDOG="${RUN_WITH_WATCHDOG:-1}"
RUN_MAX_SECONDS="${RUN_MAX_SECONDS:-3600}"
RUN_IDLE_SECONDS="${RUN_IDLE_SECONDS:-600}"
RUN_POLL_SECONDS="${RUN_POLL_SECONDS:-30}"
RUN_WATCH_PATH="${RUN_WATCH_PATH:-}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$REPO_ROOT/reports/watchdog}"
RUN_LOG_PATH="${RUN_LOG_PATH:-$RUN_LOG_DIR/run_$(date -u +%Y-%m-%dT%H-%M-%SZ).log}"
mkdir -p "$(dirname "$RUN_LOG_PATH")"

if [ "$RUN_WITH_WATCHDOG" = "1" ]; then
  WATCHDOG_ARGS=(
    --log "$RUN_LOG_PATH"
    --max-seconds "$RUN_MAX_SECONDS"
    --idle-seconds "$RUN_IDLE_SECONDS"
    --poll-seconds "$RUN_POLL_SECONDS"
  )
  if [ -n "$RUN_WATCH_PATH" ]; then
    WATCHDOG_ARGS+=(--watch-path "$RUN_WATCH_PATH")
  fi
  "$SCRIPT_DIR/run_with_activity_watchdog.sh" "${WATCHDOG_ARGS[@]}" -- "$SCRIPT_DIR/run_test_with_cleanup.sh" "$@"
else
  "$SCRIPT_DIR/run_test_with_cleanup.sh" "$@"
fi
