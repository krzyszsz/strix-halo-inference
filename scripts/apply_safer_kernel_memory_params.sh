#!/usr/bin/env bash
set -euo pipefail

GTT_MB="${GTT_MB:-65536}"
TTM_PAGES_LIMIT="${TTM_PAGES_LIMIT:-16777216}"
TTM_PAGE_POOL_SIZE="${TTM_PAGE_POOL_SIZE:-16777216}"

OLD_ARGS_REGEX='amdgpu.gttsize=[^ ]+|ttm.pages_limit=[^ ]+|ttm.page_pool_size=[^ ]+'
NEW_ARGS="amdgpu.gttsize=${GTT_MB} ttm.pages_limit=${TTM_PAGES_LIMIT} ttm.page_pool_size=${TTM_PAGE_POOL_SIZE}"

sudo_cmd() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo "$@"
  else
    "$@"
  fi
}

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

log "Current /proc/cmdline:"
cat /proc/cmdline

log "Applying safer AMD memory params to all installed kernels"
sudo_cmd grubby --update-kernel=ALL \
  --remove-args='amdgpu.gttsize=90112 ttm.pages_limit=23068672 ttm.page_pool_size=23068672' \
  --args="$NEW_ARGS"

log "Verifying kernels after update"
sudo_cmd grubby --info=ALL | rg -n 'kernel=|args=' -N

log "Done. Reboot is required for new kernel parameters to take effect."
