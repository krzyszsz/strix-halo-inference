#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

DATE_TAG="${DATE_TAG:-$(date -u +%F)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/publish/qwen35_ctx_text}"
OUT_JSON_DIR="${OUT_JSON_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/qwen35-ctx-text}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPO_ROOT/reports/publish/qwen35_ctx_text_matrix_${DATE_TAG}.tsv}"

RUN_MAX_SECONDS="${RUN_MAX_SECONDS:-2700}"
RUN_IDLE_SECONDS="${RUN_IDLE_SECONDS:-1200}"
THREADS="${THREADS:-8}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-240}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1800}"
FORCE_RERUN_FAILED="${FORCE_RERUN_FAILED:-0}"

mkdir -p "$OUT_DIR" "$OUT_JSON_DIR" "$(dirname "$SUMMARY_TSV")"

if [ ! -f "$SUMMARY_TSV" ]; then
  echo -e "ts_utc\tmodel_alias\tctx_size\tstatus\tduration_s\tport\tgpu_layers\tmem_limit\tmemory_swap\tmem_reservation\tmax_tokens\tlog_path\twatchdog_log\tartifact_json\tnotes" > "$SUMMARY_TSV"
fi

to_rel() {
  local p="$1"
  realpath --relative-to="$REPO_ROOT" "$p" 2>/dev/null || echo "$p"
}

latest_status_for_ctx() {
  local alias="$1"
  local ctx="$2"
  awk -F'\t' -v a="$alias" -v c="$ctx" 'NR>1 && $2==a && $3==c {s=$4} END {print s}' "$SUMMARY_TSV"
}

max_success_ctx() {
  local alias="$1"
  awk -F'\t' -v a="$alias" 'NR>1 && $2==a && $4=="ok" {if ($3+0 > m) m=$3+0} END {if (m>0) print m; else print 0}' "$SUMMARY_TSV"
}

append_row() {
  local alias="$1"
  local ctx="$2"
  local status="$3"
  local duration_s="$4"
  local port="$5"
  local gpu_layers="$6"
  local mem_limit="$7"
  local memory_swap="$8"
  local mem_reservation="$9"
  local max_tokens="${10}"
  local log_path="${11}"
  local watchdog_log="${12}"
  local artifact_json="${13}"
  local notes="${14}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$alias" \
    "$ctx" \
    "$status" \
    "$duration_s" \
    "$port" \
    "$gpu_layers" \
    "$mem_limit" \
    "$memory_swap" \
    "$mem_reservation" \
    "$max_tokens" \
    "$(to_rel "$log_path")" \
    "$(to_rel "$watchdog_log")" \
    "$(to_rel "$artifact_json")" \
    "$notes" \
    >> "$SUMMARY_TSV"
}

run_probe() {
  local alias="$1"
  local model_path="$2"
  local ctx="$3"
  local port="$4"
  local gpu_layers="$5"
  local mem_limit="$6"
  local memory_swap="$7"
  local mem_reservation="$8"
  local max_tokens="$9"

  local log_path="$OUT_DIR/${alias}_ctx${ctx}.log"
  local watchdog_log="$OUT_DIR/${alias}_ctx${ctx}_watchdog.log"
  local out_json="$OUT_JSON_DIR/${alias}_ctx${ctx}_${DATE_TAG}.json"

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
        MAX_TOKENS="$max_tokens" \
        PROMPT_MODE=text \
        PROMPT_TEXT="/no_think Write three short bullet points on why reproducible local LLM benchmarks matter." \
        THREADS="$THREADS" \
        GPU_LAYERS="$gpu_layers" \
        EXTRA_ARGS="--jinja --reasoning-budget 0 --reasoning-format none" \
        MEM_LIMIT="$mem_limit" \
        MEMORY_SWAP="$memory_swap" \
        MEM_RESERVATION="$mem_reservation" \
        OOM_SCORE_ADJ="$OOM_SCORE_ADJ" \
        OUT_JSON="$out_json" \
        CONTAINER="llama-qwen35-ctx-${port}" \
        READY_ATTEMPTS="$READY_ATTEMPTS" \
        CURL_MAX_TIME="$CURL_MAX_TIME" \
      bash "$REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh" \
      > "$log_path" 2>&1
  status=$?
  set -e

  end_ts="$(date +%s)"
  duration_s=$((end_ts - start_ts))

  if [ "$status" -eq 0 ] && [ -s "$out_json" ]; then
    append_row "$alias" "$ctx" "ok" "$duration_s" "$port" "$gpu_layers" "$mem_limit" "$memory_swap" "$mem_reservation" "$max_tokens" "$log_path" "$watchdog_log" "$out_json" "probe passed"
    echo "[$alias] ctx=${ctx} -> ok (${duration_s}s)"
    return 0
  fi

  append_row "$alias" "$ctx" "fail(${status})" "$duration_s" "$port" "$gpu_layers" "$mem_limit" "$memory_swap" "$mem_reservation" "$max_tokens" "$log_path" "$watchdog_log" "$out_json" "probe failed"
  echo "[$alias] ctx=${ctx} -> fail(${status}) (${duration_s}s)"
  return 1
}

CONTEXTS=(131072 65536 32768 16384 8192 4096)

MODELS=(
  "qwen35_9b_q8|$MODEL_ROOT/qwen3.5-9b-gguf/Qwen3.5-9B-Q8_0.gguf|8141|999|75g|75g|67g|256"
  "qwen35_27b_q8|$MODEL_ROOT/qwen3.5-27b-gguf/Qwen3.5-27B-Q8_0.gguf|8142|999|75g|75g|67g|256"
  "qwen35_122b_a10b_q4km|$MODEL_ROOT/qwen3.5-122b-a10b-gguf/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf|8143|0|85g|85g|76g|128"
)

for model_cfg in "${MODELS[@]}"; do
  IFS='|' read -r alias model_path port gpu_layers mem_limit memory_swap mem_reservation max_tokens <<<"$model_cfg"

  if [ ! -f "$model_path" ]; then
    append_row "$alias" "0" "skip-missing-model" "0" "$port" "$gpu_layers" "$mem_limit" "$memory_swap" "$mem_reservation" "$max_tokens" "$OUT_DIR/${alias}_missing.log" "$OUT_DIR/${alias}_missing_watchdog.log" "$OUT_JSON_DIR/${alias}_missing.json" "model file missing"
    echo "[$alias] missing model: $model_path"
    continue
  fi

  best_ok="$(max_success_ctx "$alias")"
  if [ "$best_ok" -gt 0 ]; then
    echo "[$alias] resume: existing best successful ctx=${best_ok}"
  fi

  for ctx in "${CONTEXTS[@]}"; do
    # If we already have a successful larger/equal context, skip lower ones.
    if [ "$best_ok" -ge "$ctx" ]; then
      echo "[$alias] skip ctx=${ctx} (already have successful ctx=${best_ok})"
      continue
    fi

    existing_status="$(latest_status_for_ctx "$alias" "$ctx")"
    if [ -n "$existing_status" ]; then
      if [ "$FORCE_RERUN_FAILED" = "1" ] && [[ "$existing_status" == fail* ]]; then
        echo "[$alias] rerun ctx=${ctx} (existing status=${existing_status}, FORCE_RERUN_FAILED=1)"
      else
      echo "[$alias] skip ctx=${ctx} (already recorded status=${existing_status})"
      if [ "$existing_status" = "ok" ]; then
        best_ok="$ctx"
      fi
      continue
      fi
    fi

    if run_probe "$alias" "$model_path" "$ctx" "$port" "$gpu_layers" "$mem_limit" "$memory_swap" "$mem_reservation" "$max_tokens"; then
      best_ok="$ctx"
      # Descending order: first success is max among requested levels.
      break
    fi
  done

  echo "[$alias] current max successful ctx among requested levels: ${best_ok}"
done

echo
echo "Done. Summary: $(to_rel "$SUMMARY_TSV")"
echo "Per-run logs: $(to_rel "$OUT_DIR")"
