#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

DATE_TAG="${DATE_TAG:-$(date -u +%F)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/publish/gemma4_ctx_text}"
OUT_JSON_DIR="${OUT_JSON_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/gemma4-ctx-text}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPO_ROOT/reports/publish/gemma4_ctx_text_matrix_${DATE_TAG}.tsv}"

RUN_MAX_SECONDS="${RUN_MAX_SECONDS:-3600}"
RUN_IDLE_SECONDS="${RUN_IDLE_SECONDS:-1200}"
THREADS="${THREADS:-8}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-240}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1800}"
PORT_BASE="${PORT_BASE:-8160}"
MAX_TOKENS="${MAX_TOKENS:-256}"

MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"

MODEL_PRIMARY="${MODEL_PRIMARY:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q8_0.gguf}"
MODEL_FALLBACK="${MODEL_FALLBACK:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf}"
PRIMARY_LABEL="${PRIMARY_LABEL:-primary}"
FALLBACK_LABEL="${FALLBACK_LABEL:-fallback}"

CONTEXTS=(32768 65536 131072 172032 262144)

mkdir -p "$OUT_DIR" "$OUT_JSON_DIR" "$(dirname "$SUMMARY_TSV")"

if [ ! -f "$SUMMARY_TSV" ]; then
  echo -e "ts_utc\tctx_size\tstatus\tattempt_label\tduration_s\tport\tgpu_layers\tmem_limit\tmemory_swap\tmem_reservation\tmax_tokens\tlog_path\twatchdog_log\tartifact_json\tnotes" > "$SUMMARY_TSV"
fi

to_rel() {
  local p="$1"
  realpath --relative-to="$REPO_ROOT" "$p" 2>/dev/null || echo "$p"
}

append_row() {
  local ctx="$1"
  local status="$2"
  local attempt_label="$3"
  local duration_s="$4"
  local port="$5"
  local gpu_layers="$6"
  local log_path="$7"
  local watchdog_log="$8"
  local artifact_json="$9"
  local notes="${10}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$ctx" \
    "$status" \
    "$attempt_label" \
    "$duration_s" \
    "$port" \
    "$gpu_layers" \
    "$MEM_LIMIT" \
    "$MEMORY_SWAP" \
    "$MEM_RESERVATION" \
    "$MAX_TOKENS" \
    "$(to_rel "$log_path")" \
    "$(to_rel "$watchdog_log")" \
    "$(to_rel "$artifact_json")" \
    "$notes" \
    >> "$SUMMARY_TSV"
}

run_attempt() {
  local ctx="$1"
  local model_path="$2"
  local gpu_layers="$3"
  local attempt_label="$4"
  local port="$5"

  local log_path="$OUT_DIR/gemma4_ctx${ctx}_${attempt_label}.log"
  local watchdog_log="$OUT_DIR/gemma4_ctx${ctx}_${attempt_label}_watchdog.log"
  local out_json="$OUT_JSON_DIR/gemma4_ctx${ctx}_${attempt_label}_${DATE_TAG}.json"

  local start_ts end_ts duration_s status
  start_ts="$(date +%s)"
  set +e
  RUN_WITH_WATCHDOG=1 \
  RUN_MAX_SECONDS="$RUN_MAX_SECONDS" \
  RUN_IDLE_SECONDS="$RUN_IDLE_SECONDS" \
  RUN_LOG_PATH="$watchdog_log" \
    "$REPO_ROOT/scripts/run_memsafe.sh" \
      env \
        MODEL_PATH="$model_path" \
        PORT="$port" \
        CTX_SIZE="$ctx" \
        MAX_TOKENS="$MAX_TOKENS" \
        PROMPT_MODE=text \
        PROMPT_TEXT="/no_think Write 5 concise bullets explaining why memory-safe local inference harnesses matter." \
        THREADS="$THREADS" \
        GPU_LAYERS="$gpu_layers" \
        EXTRA_ARGS="--jinja --reasoning-budget 0 --reasoning-format none" \
        MEM_LIMIT="$MEM_LIMIT" \
        MEMORY_SWAP="$MEMORY_SWAP" \
        MEM_RESERVATION="$MEM_RESERVATION" \
        OOM_SCORE_ADJ="$OOM_SCORE_ADJ" \
        OUT_JSON="$out_json" \
        CONTAINER="llama-gemma4-ctx-${port}" \
        READY_ATTEMPTS="$READY_ATTEMPTS" \
        CURL_MAX_TIME="$CURL_MAX_TIME" \
      bash "$REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh" \
      > "$log_path" 2>&1
  status=$?
  set -e
  end_ts="$(date +%s)"
  duration_s=$((end_ts - start_ts))

  if [ "$status" -eq 0 ] && [ -s "$out_json" ]; then
    append_row "$ctx" "ok" "$attempt_label" "$duration_s" "$port" "$gpu_layers" "$log_path" "$watchdog_log" "$out_json" "probe passed"
    echo "[ctx=${ctx}] ${attempt_label} -> ok (${duration_s}s)"
    return 0
  fi

  append_row "$ctx" "fail(${status})" "$attempt_label" "$duration_s" "$port" "$gpu_layers" "$log_path" "$watchdog_log" "$out_json" "probe failed"
  echo "[ctx=${ctx}] ${attempt_label} -> fail(${status}) (${duration_s}s)"
  return 1
}

if [ ! -f "$MODEL_PRIMARY" ]; then
  echo "Missing primary model: $MODEL_PRIMARY" >&2
  exit 2
fi

if [ ! -f "$MODEL_FALLBACK" ]; then
  echo "Warning: fallback model missing: $MODEL_FALLBACK" >&2
fi

for idx in "${!CONTEXTS[@]}"; do
  ctx="${CONTEXTS[$idx]}"
  port=$((PORT_BASE + idx))
  echo "== ctx ${ctx} =="

  if run_attempt "$ctx" "$MODEL_PRIMARY" 999 "${PRIMARY_LABEL}_gpu" "$port"; then
    continue
  fi

  if run_attempt "$ctx" "$MODEL_PRIMARY" 0 "${PRIMARY_LABEL}_cpu" "$port"; then
    continue
  fi

  if [ -f "$MODEL_FALLBACK" ] && [ "$MODEL_FALLBACK" != "$MODEL_PRIMARY" ]; then
    if run_attempt "$ctx" "$MODEL_FALLBACK" 999 "${FALLBACK_LABEL}_gpu" "$port"; then
      continue
    fi
    run_attempt "$ctx" "$MODEL_FALLBACK" 0 "${FALLBACK_LABEL}_cpu" "$port" || true
  fi
done

echo
echo "Done. Summary: $(to_rel "$SUMMARY_TSV")"
echo "Logs: $(to_rel "$OUT_DIR")"
