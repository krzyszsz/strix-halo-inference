#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT="${OUT:-$REPO_ROOT/reports/verify_hf_sha256.json}"
mkdir -p "$(dirname "$OUT")"

"$REPO_ROOT/scripts/verify_hf_sha256.py" > "$OUT"

echo "Wrote $OUT"
