#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/cleanup}"
mkdir -p "$REPORT_DIR"

LOCK_MIN_AGE_SECONDS="${LOCK_MIN_AGE_SECONDS:-900}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_PATH="$REPORT_DIR/clear_hf_locks_${TIMESTAMP}.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_PATH"
}

rm_cmd() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo rm -f "$1" >/dev/null 2>&1 || true
  else
    rm -f "$1" >/dev/null 2>&1 || true
  fi
}

if [ ! -d "$MODEL_ROOT" ]; then
  log "MODEL_ROOT does not exist, skipping lock cleanup: $MODEL_ROOT"
  echo "$LOG_PATH"
  exit 0
fi

log "Scanning stale HF lock files under $MODEL_ROOT (min_age=${LOCK_MIN_AGE_SECONDS}s)"
now_ts="$(date +%s)"
removed=0
kept_recent=0
kept_open=0

while IFS= read -r -d '' lock_path; do
  mtime="$(stat -c %Y "$lock_path" 2>/dev/null || echo 0)"
  age=$((now_ts - mtime))
  if [ "$age" -lt "$LOCK_MIN_AGE_SECONDS" ]; then
    kept_recent=$((kept_recent + 1))
    continue
  fi

  if command -v lsof >/dev/null 2>&1; then
    if lsof "$lock_path" >/dev/null 2>&1; then
      kept_open=$((kept_open + 1))
      continue
    fi
  fi

  log "Removing stale lock: $lock_path (age=${age}s)"
  rm_cmd "$lock_path"
  removed=$((removed + 1))
done < <(find "$MODEL_ROOT" -type f -name '*.lock' -print0 2>/dev/null)

log "Done stale lock cleanup removed=$removed kept_recent=$kept_recent kept_open=$kept_open log=$LOG_PATH"
echo "$LOG_PATH"
