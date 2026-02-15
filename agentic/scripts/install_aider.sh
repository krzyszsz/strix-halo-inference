#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if command -v python3.12 >/dev/null 2>&1; then
  python3.12 -m pip install --upgrade aider-chat
  exit 0
fi

if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m pip install --upgrade aider-chat
  exit 0
fi

if command -v python3 >/dev/null 2>&1; then
  python3 -m pip install --upgrade aider-chat
  exit 0
fi

echo "No supported Python (3.12/3.11) found for local install. Use the containerized script instead:" >&2
echo "  $REPO_ROOT/agentic/scripts/run_aider_container.sh" >&2
exit 1
