#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE' >&2
Usage:
  run_with_activity_watchdog.sh --log <path> --max-seconds <n> --idle-seconds <n> [--watch-path <path>] [--poll-seconds <n>] [--heartbeat-seconds <n>] [--min-available-gb <n>] [--max-swap-used-gb <n>] -- <command...>

Example:
  run_with_activity_watchdog.sh \
    --log reports/example.log \
    --max-seconds 3600 \
    --idle-seconds 600 \
    --watch-path /mnt/hf/models/my-model \
    -- bash my_script.sh
USAGE
}

LOG_PATH=""
MAX_SECONDS=""
IDLE_SECONDS=""
WATCH_PATH=""
POLL_SECONDS=30
HEARTBEAT_SECONDS=120
MIN_AVAILABLE_GB="${MIN_AVAILABLE_GB:-16}"
MAX_SWAP_USED_GB="${MAX_SWAP_USED_GB:-24}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --log)
      LOG_PATH="${2:-}"
      shift 2
      ;;
    --max-seconds)
      MAX_SECONDS="${2:-}"
      shift 2
      ;;
    --idle-seconds)
      IDLE_SECONDS="${2:-}"
      shift 2
      ;;
    --watch-path)
      WATCH_PATH="${2:-}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:-}"
      shift 2
      ;;
    --heartbeat-seconds)
      HEARTBEAT_SECONDS="${2:-}"
      shift 2
      ;;
    --min-available-gb)
      MIN_AVAILABLE_GB="${2:-}"
      shift 2
      ;;
    --max-swap-used-gb)
      MAX_SWAP_USED_GB="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$LOG_PATH" ] || [ -z "$MAX_SECONDS" ] || [ -z "$IDLE_SECONDS" ] || [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

mkdir -p "$(dirname "$LOG_PATH")"
touch "$LOG_PATH"

ts() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

mem_kb() {
  local key="$1"
  awk -v k="$key" '$1 == (k ":") {print $2}' /proc/meminfo
}

to_gb_rounded() {
  local kb="$1"
  awk -v kb="$kb" 'BEGIN {printf "%.1f", kb / 1024 / 1024}'
}

sanitize_cmd() {
  local raw="$1"
  printf '%s' "$raw" | sed -E \
    -e 's/hf_[A-Za-z0-9]{10,}/hf_REDACTED/g' \
    -e 's/(HF_TOKEN=)[^ ]+/\1REDACTED/g' \
    -e 's/(HUGGINGFACE_HUB_TOKEN=)[^ ]+/\1REDACTED/g' \
    -e 's/(Authorization: Bearer )[A-Za-z0-9._-]+/\1REDACTED/g'
}

raw_cmd="$*"
safe_cmd="$(sanitize_cmd "$raw_cmd")"
echo "[$(ts)] watchdog start max_seconds=$MAX_SECONDS idle_seconds=$IDLE_SECONDS poll_seconds=$POLL_SECONDS watch_path=${WATCH_PATH:-<none>} min_available_gb=$MIN_AVAILABLE_GB max_swap_used_gb=$MAX_SWAP_USED_GB cmd=$safe_cmd" | tee -a "$LOG_PATH"
echo "watchdog: log=$LOG_PATH" >&2

set +e
cmd_pgid=""
if command -v setsid >/dev/null 2>&1; then
  setsid "$@" >>"$LOG_PATH" 2>&1 &
  cmd_pid=$!
  cmd_pgid="$cmd_pid"
else
  "$@" >>"$LOG_PATH" 2>&1 &
  cmd_pid=$!
fi
set -e

start_ts="$(date +%s)"
last_change_ts="$start_ts"
last_heartbeat_ts="$start_ts"
last_size="$(stat -c%s "$LOG_PATH" 2>/dev/null || echo 0)"
last_watch_size=0
if [ -n "$WATCH_PATH" ]; then
  if [ -e "$WATCH_PATH" ]; then
    last_watch_size="$(du -sb "$WATCH_PATH" 2>/dev/null | awk '{print $1}' || echo 0)"
  fi
fi
watchdog_status=0

run_repo_cleanup() {
  # Best-effort: avoid leaving behind runaway containers if we abort a run.
  if [ -x "$SCRIPT_DIR/cleanup_machine.sh" ]; then
    "$SCRIPT_DIR/cleanup_machine.sh" >/dev/null 2>&1 || true
  fi
}

terminate_cmd() {
  local signal="$1"
  if [ -n "$cmd_pgid" ]; then
    kill "-$signal" -- "-$cmd_pgid" >/dev/null 2>&1 || true
  fi
  kill "-$signal" "$cmd_pid" >/dev/null 2>&1 || true
}

while kill -0 "$cmd_pid" >/dev/null 2>&1; do
  sleep "$POLL_SECONDS"
  now_ts="$(date +%s)"
  now_size="$(stat -c%s "$LOG_PATH" 2>/dev/null || echo 0)"

  if [ "$now_size" -gt "$last_size" ]; then
    last_size="$now_size"
    last_change_ts="$now_ts"
  fi

  if [ -n "$WATCH_PATH" ] && [ -e "$WATCH_PATH" ]; then
    now_watch_size="$(du -sb "$WATCH_PATH" 2>/dev/null | awk '{print $1}' || echo 0)"
    if [ "$now_watch_size" -ne "$last_watch_size" ]; then
      last_watch_size="$now_watch_size"
      last_change_ts="$now_ts"
    fi
  fi

  elapsed=$((now_ts - start_ts))
  idle_elapsed=$((now_ts - last_change_ts))

  # Host safety guard: if memory is getting dangerously low or swap is heavily used,
  # abort the run before the OS becomes unresponsive.
  if [ "${MIN_AVAILABLE_GB:-0}" -gt 0 ] || [ "${MAX_SWAP_USED_GB:-0}" -gt 0 ]; then
    mem_available_kb="$(mem_kb MemAvailable || true)"
    mem_available_kb="${mem_available_kb:-0}"
    swap_total_kb="$(mem_kb SwapTotal || true)"
    swap_total_kb="${swap_total_kb:-0}"
    swap_free_kb="$(mem_kb SwapFree || true)"
    swap_free_kb="${swap_free_kb:-0}"
    swap_used_kb=$((swap_total_kb - swap_free_kb))

    if [ "${MIN_AVAILABLE_GB:-0}" -gt 0 ]; then
      min_available_kb=$((MIN_AVAILABLE_GB * 1024 * 1024))
      if [ "$mem_available_kb" -lt "$min_available_kb" ]; then
        echo "[$(ts)] watchdog host-memory guard tripped: MemAvailable=$(to_gb_rounded "$mem_available_kb")GiB < ${MIN_AVAILABLE_GB}GiB; terminating pid=$cmd_pid" | tee -a "$LOG_PATH"
        terminate_cmd TERM
        sleep 5
        terminate_cmd KILL
        run_repo_cleanup
        watchdog_status=126
        break
      fi
    fi

    if [ "${MAX_SWAP_USED_GB:-0}" -gt 0 ]; then
      max_swap_used_kb=$((MAX_SWAP_USED_GB * 1024 * 1024))
      if [ "$swap_used_kb" -gt "$max_swap_used_kb" ]; then
        echo "[$(ts)] watchdog host-swap guard tripped: SwapUsed=$(to_gb_rounded "$swap_used_kb")GiB > ${MAX_SWAP_USED_GB}GiB; terminating pid=$cmd_pid" | tee -a "$LOG_PATH"
        terminate_cmd TERM
        sleep 5
        terminate_cmd KILL
        run_repo_cleanup
        watchdog_status=127
        break
      fi
    fi
  fi

  if [ "$HEARTBEAT_SECONDS" -gt 0 ] && [ $((now_ts - last_heartbeat_ts)) -ge "$HEARTBEAT_SECONDS" ]; then
    last_heartbeat_ts="$now_ts"
    echo "watchdog: running elapsed=${elapsed}s idle=${idle_elapsed}s log=$LOG_PATH" >&2
  fi

  if [ "$elapsed" -ge "$MAX_SECONDS" ]; then
    echo "[$(ts)] watchdog timeout reached (elapsed=${elapsed}s), terminating pid=$cmd_pid" | tee -a "$LOG_PATH"
    terminate_cmd TERM
    sleep 5
    terminate_cmd KILL
    run_repo_cleanup
    watchdog_status=124
    break
  fi

  if [ "$idle_elapsed" -ge "$IDLE_SECONDS" ]; then
    echo "[$(ts)] watchdog idle timeout reached (idle=${idle_elapsed}s), terminating pid=$cmd_pid" | tee -a "$LOG_PATH"
    terminate_cmd TERM
    sleep 5
    terminate_cmd KILL
    run_repo_cleanup
    watchdog_status=125
    break
  fi
done

if [ "$watchdog_status" -ne 0 ]; then
  wait "$cmd_pid" >/dev/null 2>&1 || true
  echo "[$(ts)] watchdog exit status=$watchdog_status" | tee -a "$LOG_PATH"
  exit "$watchdog_status"
fi

set +e
wait "$cmd_pid"
cmd_status=$?
set -e

echo "[$(ts)] watchdog finished cmd_status=$cmd_status" | tee -a "$LOG_PATH"
exit "$cmd_status"
