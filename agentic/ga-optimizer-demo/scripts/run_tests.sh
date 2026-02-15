#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SLN="$REPO_ROOT/agentic/ga-optimizer-demo/GaOptimizerDemo.slnx"
OUT_LOG="${OUT_LOG:-$REPO_ROOT/agentic/ga-optimizer-demo/out/tests_post_impl_recheck_publish.log}"
mkdir -p "$(dirname "$OUT_LOG")"

# Some publish-time cleanups can delete NuGet package folders while leaving assets files in place.
# Force restore to avoid confusing "package not found" errors.
dotnet restore -f --no-http-cache "$SLN"

dotnet test --no-restore "$SLN" | tee "$OUT_LOG"
