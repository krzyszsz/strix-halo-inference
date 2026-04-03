#!/usr/bin/env bash
set -euo pipefail

SWAP_PATH="${SWAP_PATH:-/var/swap/strix-halo.swap}"
SWAP_SIZE_GB="${SWAP_SIZE_GB:-64}"
SWAP_PRIO="${SWAP_PRIO:-10}"
FSTAB_TAG="# strix-halo-inference-managed"

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

ensure_swapfile() {
  if [ -f "$SWAP_PATH" ]; then
    log "Swapfile already exists: $SWAP_PATH"
    return
  fi

  local swap_dir
  swap_dir="$(dirname "$SWAP_PATH")"
  sudo_cmd mkdir -p "$swap_dir"

  local fstype
  fstype="$(findmnt -no FSTYPE "$swap_dir" || findmnt -no FSTYPE /)"

  if [ "$fstype" = "btrfs" ] && command -v btrfs >/dev/null 2>&1; then
    log "Creating Btrfs swapfile (${SWAP_SIZE_GB}G) at $SWAP_PATH"
    sudo_cmd btrfs filesystem mkswapfile --size "${SWAP_SIZE_GB}g" "$SWAP_PATH"
  else
    log "Creating swapfile (${SWAP_SIZE_GB}G) at $SWAP_PATH"
    sudo_cmd fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_PATH"
    sudo_cmd chmod 600 "$SWAP_PATH"
    sudo_cmd mkswap "$SWAP_PATH" >/dev/null
  fi
}

ensure_swapon() {
  if swapon --show=NAME | rg -Fxq "$SWAP_PATH"; then
    log "Swapfile already active: $SWAP_PATH"
  else
    log "Activating swapfile: $SWAP_PATH (priority ${SWAP_PRIO})"
    sudo_cmd swapon --priority "$SWAP_PRIO" "$SWAP_PATH"
  fi
}

ensure_fstab() {
  local entry="${SWAP_PATH} none swap defaults,pri=${SWAP_PRIO} 0 0 ${FSTAB_TAG}"
  if sudo_cmd grep -Fq "$SWAP_PATH" /etc/fstab; then
    log "Swapfile already present in /etc/fstab"
  else
    log "Persisting swapfile in /etc/fstab"
    echo "$entry" | sudo_cmd tee -a /etc/fstab >/dev/null
  fi
}

log "Configuring additional swap safety"
ensure_swapfile
ensure_swapon
ensure_fstab
log "Swap status:"
swapon --show
