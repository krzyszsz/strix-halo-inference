#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/cleanup}"
mkdir -p "$REPORT_DIR"

# Threshold in MB for process RSS cleanup.
RSS_MB_THRESHOLD="${RSS_MB_THRESHOLD:-5120}"
RSS_KB_THRESHOLD=$((RSS_MB_THRESHOLD * 1024))
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_PATH="$REPORT_DIR/cleanup_${TIMESTAMP}.log"
CONTAINER_SAFE_REGEX="${CONTAINER_SAFE_REGEX:-^$}"

# Safe-list to avoid killing desktop/editor/agent infrastructure.
SAFE_COMM_REGEX='^(code|codex|gnome-shell|Xwayland|Xorg|systemd|containerd|dockerd|dbus-daemon|pipewire|wireplumber|ptyxis)$'

docker_cmd() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_PATH"
}

kill_large_processes() {
  log "Scanning processes above ${RSS_MB_THRESHOLD}MB RSS"
  local self_pid="$$"
  local parent_pid="$PPID"

  # pid user rss comm args
  while read -r pid user rss comm args; do
    [ -z "${pid:-}" ] && continue
    if [ "$rss" -lt "$RSS_KB_THRESHOLD" ]; then
      continue
    fi

    if [ "$pid" = "$self_pid" ] || [ "$pid" = "$parent_pid" ]; then
      continue
    fi

    if [[ "$comm" =~ $SAFE_COMM_REGEX ]]; then
      log "Skipping safe process pid=$pid comm=$comm rss_kb=$rss"
      continue
    fi

    log "Terminating pid=$pid user=$user comm=$comm rss_kb=$rss"
    if [ "$(id -u)" -ne 0 ]; then
      sudo kill -TERM "$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi

    sleep 2

    if ps -p "$pid" >/dev/null 2>&1; then
      log "Force-killing pid=$pid"
      if [ "$(id -u)" -ne 0 ]; then
        sudo kill -KILL "$pid" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  done < <(ps -eo pid,user,rss,comm,args --sort=-rss | tail -n +2)
}

cleanup_test_containers() {
  log "Cleaning up inference/test containers"

  # Remove all exited containers first.
  local exited_ids
  exited_ids="$(docker_cmd ps -aq -f status=exited || true)"
  if [ -n "$exited_ids" ]; then
    docker_cmd rm -f $exited_ids >/dev/null 2>&1 || true
    log "Removed exited containers"
  fi

  # Stop/remove active containers unless explicitly safelisted.
  local running
  running="$(docker_cmd ps --format '{{.ID}} {{.Names}} {{.Image}}')"
  if [ -n "$running" ]; then
    while read -r cid cname cimage; do
      [ -z "${cid:-}" ] && continue
      if [[ "$cname $cimage" =~ $CONTAINER_SAFE_REGEX ]]; then
        log "Skipping safelisted container name=$cname image=$cimage"
      else
        log "Removing active container name=$cname image=$cimage"
        docker_cmd rm -f "$cid" >/dev/null 2>&1 || true
      fi
    done <<< "$running"
  fi
}

log "Cleanup start"
kill_large_processes
cleanup_test_containers
log "Memory snapshot after cleanup:"
free -h | tee -a "$LOG_PATH" >/dev/null
log "Active swap devices after cleanup:"
swapon --show | tee -a "$LOG_PATH" >/dev/null

log "Top memory consumers after cleanup:"
ps -eo pid,user,rss,comm,args --sort=-rss | head -n 20 | tee -a "$LOG_PATH" >/dev/null

log "Cleanup done; log=$LOG_PATH"
echo "$LOG_PATH"
