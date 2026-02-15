#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_ALIAS="${MODEL_ALIAS:?MODEL_ALIAS is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
PORT="${PORT:-8103}"
PROMPT_MODE="${PROMPT_MODE:-text}"
MIN_CTX="${MIN_CTX:-2048}"
MAX_CTX_CAP="${MAX_CTX_CAP:-131072}"
PRECISION="${PRECISION:-1024}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-999}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/reports/retest_2026-02-10_ctx}"

RUN_DIR="$RUN_ROOT/$MODEL_ALIAS"
ATTEMPTS_TSV="$RUN_DIR/attempts.tsv"
RESULT_TXT="$RUN_DIR/result.txt"
mkdir -p "$RUN_DIR"
mkdir -p "$REPO_ROOT/llama-cpp-vulkan/out"

if [ ! -f "$ATTEMPTS_TSV" ]; then
  echo -e "ts_utc\tmodel_alias\tctx_size\tmax_tokens\tstatus\tduration_s\tlog_path\tartifact_path\tnotes" > "$ATTEMPTS_TSV"
fi

declare -A PROBE_CACHE

to_k_step() {
  local value="$1"
  local step="$2"
  echo $(( (value / step) * step ))
}

min_ctx_rounded=$(( ((MIN_CTX + PRECISION - 1) / PRECISION) * PRECISION ))
cap_ctx_rounded="$(to_k_step "$MAX_CTX_CAP" "$PRECISION")"

if [ "$cap_ctx_rounded" -lt "$min_ctx_rounded" ]; then
  cap_ctx_rounded="$min_ctx_rounded"
fi

run_probe() {
  local ctx="$1"
  if [ -n "${PROBE_CACHE[$ctx]+x}" ]; then
    if [ "${PROBE_CACHE[$ctx]}" = "ok" ]; then
      return 0
    fi
    return 1
  fi

  local log_path="$RUN_DIR/ctx_${ctx}.log"
  local out_json="$REPO_ROOT/llama-cpp-vulkan/out/${MODEL_ALIAS}_ctx_${ctx}.json"
  local start_ts end_ts dur status ts_utc
  start_ts="$(date +%s)"
  ts_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  set +e
  "$REPO_ROOT/scripts/run_test_with_cleanup.sh" \
    env \
      MODEL_PATH="$MODEL_PATH" \
      PORT="$PORT" \
      CTX_SIZE="$ctx" \
      MAX_TOKENS="$MAX_TOKENS" \
      PROMPT_MODE="$PROMPT_MODE" \
      THREADS="$THREADS" \
      GPU_LAYERS="$GPU_LAYERS" \
      MEM_LIMIT="$MEM_LIMIT" \
      MEMORY_SWAP="$MEMORY_SWAP" \
      MEM_RESERVATION="$MEM_RESERVATION" \
      OOM_SCORE_ADJ="$OOM_SCORE_ADJ" \
      OUT_JSON="$out_json" \
      CONTAINER="llama-ctx-probe-${PORT}" \
      bash "$SCRIPT_DIR/probe_ctx_once.sh" \
      >"$log_path" 2>&1
  status=$?
  set -e

  end_ts="$(date +%s)"
  dur=$((end_ts - start_ts))

  if [ "$status" -eq 0 ]; then
    PROBE_CACHE["$ctx"]="ok"
    echo -e "${ts_utc}\t${MODEL_ALIAS}\t${ctx}\t${MAX_TOKENS}\tok\t${dur}\t${log_path}\t${out_json}\tprobe passed" >> "$ATTEMPTS_TSV"
    return 0
  fi

  PROBE_CACHE["$ctx"]="fail"
  echo -e "${ts_utc}\t${MODEL_ALIAS}\t${ctx}\t${MAX_TOKENS}\tfail(${status})\t${dur}\t${log_path}\t${out_json}\tprobe failed" >> "$ATTEMPTS_TSV"
  return 1
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] model=${MODEL_ALIAS} min=${min_ctx_rounded} cap=${cap_ctx_rounded} precision=${PRECISION} max_tokens=${MAX_TOKENS}"

if ! run_probe "$min_ctx_rounded"; then
  {
    echo "model_alias=${MODEL_ALIAS}"
    echo "model_path=${MODEL_PATH}"
    echo "prompt_mode=${PROMPT_MODE}"
    echo "max_tokens=${MAX_TOKENS}"
    echo "precision=${PRECISION}"
    echo "max_stable_ctx=0"
    echo "first_failed_ctx=${min_ctx_rounded}"
    echo "notes=minimum context probe failed"
  } > "$RESULT_TXT"
  echo "Failed at minimum context ${min_ctx_rounded}; stopping search."
  exit 1
fi

best_ok="$min_ctx_rounded"
first_fail="-1"
next_ctx=$((best_ok * 2))

while [ "$next_ctx" -le "$cap_ctx_rounded" ]; do
  next_ctx="$(to_k_step "$next_ctx" "$PRECISION")"
  if [ "$next_ctx" -le "$best_ok" ]; then
    next_ctx=$((best_ok + PRECISION))
  fi

  if run_probe "$next_ctx"; then
    best_ok="$next_ctx"
    next_ctx=$((next_ctx * 2))
    continue
  fi

  first_fail="$next_ctx"
  break
done

if [ "$first_fail" = "-1" ]; then
  if [ "$best_ok" -lt "$cap_ctx_rounded" ]; then
    if run_probe "$cap_ctx_rounded"; then
      best_ok="$cap_ctx_rounded"
      first_fail=$((cap_ctx_rounded + PRECISION))
    else
      first_fail="$cap_ctx_rounded"
    fi
  else
    first_fail=$((cap_ctx_rounded + PRECISION))
  fi
fi

while [ $((first_fail - best_ok)) -gt "$PRECISION" ]; do
  mid=$(( (best_ok + first_fail) / 2 ))
  mid="$(to_k_step "$mid" "$PRECISION")"

  if [ "$mid" -le "$best_ok" ]; then
    mid=$((best_ok + PRECISION))
  fi
  if [ "$mid" -ge "$first_fail" ]; then
    mid=$((first_fail - PRECISION))
  fi

  if run_probe "$mid"; then
    best_ok="$mid"
  else
    first_fail="$mid"
  fi
done

attempt_count=$(( $(wc -l < "$ATTEMPTS_TSV") - 1 ))

{
  echo "model_alias=${MODEL_ALIAS}"
  echo "model_path=${MODEL_PATH}"
  echo "prompt_mode=${PROMPT_MODE}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "precision=${PRECISION}"
  echo "max_stable_ctx=${best_ok}"
  echo "first_failed_ctx=${first_fail}"
  echo "attempt_count=${attempt_count}"
  echo "attempts_tsv=${ATTEMPTS_TSV}"
} > "$RESULT_TXT"

SUMMARY_TSV="$RUN_ROOT/summary.tsv"
if [ ! -f "$SUMMARY_TSV" ]; then
  echo -e "ts_utc\tmodel_alias\tprompt_mode\tmax_stable_ctx\tfirst_failed_ctx\tprecision\tmax_tokens\tattempt_count\tresult_path" > "$SUMMARY_TSV"
fi
echo -e "$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${MODEL_ALIAS}\t${PROMPT_MODE}\t${best_ok}\t${first_fail}\t${PRECISION}\t${MAX_TOKENS}\t${attempt_count}\t${RESULT_TXT}" >> "$SUMMARY_TSV"

echo "Search completed for ${MODEL_ALIAS}: max_stable_ctx=${best_ok}, first_failed_ctx=${first_fail}"
