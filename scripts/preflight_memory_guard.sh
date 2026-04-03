#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/cleanup}"
mkdir -p "$REPORT_DIR"

MIN_AVAILABLE_GB="${MIN_AVAILABLE_GB:-16}"
MAX_SWAP_USED_GB="${MAX_SWAP_USED_GB:-24}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_PATH="$REPORT_DIR/preflight_${TIMESTAMP}.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_PATH"
}

mem_kb() {
  local key="$1"
  awk -v k="$key" '$1 == (k ":") {print $2}' /proc/meminfo
}

to_gb_rounded() {
  local kb="$1"
  awk -v kb="$kb" 'BEGIN {printf "%.1f", kb / 1024 / 1024}'
}

mem_total_kb="$(mem_kb MemTotal)"
mem_available_kb="$(mem_kb MemAvailable)"
swap_total_kb="$(mem_kb SwapTotal)"
swap_free_kb="$(mem_kb SwapFree)"
swap_used_kb=$((swap_total_kb - swap_free_kb))

min_available_kb=$((MIN_AVAILABLE_GB * 1024 * 1024))
max_swap_used_kb=$((MAX_SWAP_USED_GB * 1024 * 1024))

log "Preflight start"
log "MemTotal=$(to_gb_rounded "$mem_total_kb")GiB MemAvailable=$(to_gb_rounded "$mem_available_kb")GiB"
log "SwapTotal=$(to_gb_rounded "$swap_total_kb")GiB SwapUsed=$(to_gb_rounded "$swap_used_kb")GiB"
log "Thresholds: MIN_AVAILABLE_GB=${MIN_AVAILABLE_GB}, MAX_SWAP_USED_GB=${MAX_SWAP_USED_GB}"

if [ "$mem_available_kb" -lt "$min_available_kb" ]; then
  log "MemAvailable below threshold; aborting this run to avoid host memory deadlock."
  exit 70
fi

if [ "$swap_used_kb" -gt "$max_swap_used_kb" ]; then
  log "SwapUsed above threshold; aborting this run to avoid heavy swap thrash."
  exit 71
fi

log "Preflight passed"
echo "$LOG_PATH"
