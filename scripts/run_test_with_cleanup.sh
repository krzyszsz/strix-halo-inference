#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command ...>" >&2
  exit 1
fi

run_cleanup_pass() {
  "$SCRIPT_DIR/cleanup_machine.sh" >/dev/null || true
  "$SCRIPT_DIR/preflight_memory_guard.sh" >/dev/null || true
  if [ "${CLEAR_HF_LOCKS:-1}" = "1" ]; then
    "$SCRIPT_DIR/clear_hf_locks.sh" >/dev/null || true
  fi
}

run_cleanup_pass

post_run_cleanup() {
  run_cleanup_pass
}
trap post_run_cleanup EXIT INT TERM

set +e
"$@"
status=$?
set -e

exit "$status"
