#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

docker() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

DATE_TAG="${DATE_TAG:-$(date -u +%F)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/publish/text_bench_${DATE_TAG}}"
OUT_JSON_DIR="${OUT_JSON_DIR:-$REPO_ROOT/llama-cpp-vulkan/out/text-bench-${DATE_TAG}}"
SUMMARY_TSV="${SUMMARY_TSV:-$REPO_ROOT/reports/publish/text_model_speed_bench_${DATE_TAG}.tsv}"
CONTEXTS_CSV="${CONTEXTS_CSV:-16384,32768,46080,65536,76800}"
TASKS_CSV="${TASKS_CSV:-coding_api,coding_review,summary_report,summary_plan}"
MAX_TOKENS="${MAX_TOKENS:-384}"
THREADS="${THREADS:-8}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
READY_ATTEMPTS="${READY_ATTEMPTS:-240}"
READY_SLEEP_SECS="${READY_SLEEP_SECS:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-1800}"
RUN_MAX_SECONDS="${RUN_MAX_SECONDS:-3600}"
RUN_IDLE_SECONDS="${RUN_IDLE_SECONDS:-900}"
RUN_POLL_SECONDS="${RUN_POLL_SECONDS:-30}"
LONG_CONTEXT_FILL_RATIO="${LONG_CONTEXT_FILL_RATIO:-0.55}"
LLAMA_IMAGE="${LLAMA_IMAGE:-llama-cpp-vulkan:latest}"
MODEL_FILTER="${MODEL_FILTER:-}"

mkdir -p "$OUT_DIR" "$OUT_JSON_DIR" "$(dirname "$SUMMARY_TSV")"

if [ ! -f "$SUMMARY_TSV" ]; then
  printf 'ts_utc\tmodel_alias\tctx_size\ttask\tstatus\tduration_s\tprompt_chars\tcompletion_chars\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tport\tgpu_layers\tmem_limit\tjson_path\tlog_path\tnotes\n' > "$SUMMARY_TSV"
fi

to_rel() {
  local p="$1"
  realpath --relative-to="$REPO_ROOT" "$p" 2>/dev/null || echo "$p"
}

recorded_status() {
  local alias="$1"
  local ctx="$2"
  local task="$3"
  awk -F'\t' -v a="$alias" -v c="$ctx" -v t="$task" 'NR>1 && $2==a && $3==c && $4==t {s=$5} END {print s}' "$SUMMARY_TSV"
}

all_tasks_recorded() {
  local alias="$1"
  local ctx="$2"
  shift 2
  local task
  for task in "$@"; do
    if [ -z "$(recorded_status "$alias" "$ctx" "$task")" ]; then
      return 1
    fi
  done
  return 0
}

append_row() {
  local alias="$1" ctx="$2" task="$3" status="$4" duration_s="$5" prompt_chars="$6" completion_chars="$7"
  local prompt_tokens="$8" completion_tokens="$9" total_tokens="${10}" port="${11}" gpu_layers="${12}" mem_limit="${13}"
  local json_path="${14}" log_path="${15}" notes="${16}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$alias" "$ctx" "$task" "$status" "$duration_s" \
    "$prompt_chars" "$completion_chars" "$prompt_tokens" "$completion_tokens" "$total_tokens" \
    "$port" "$gpu_layers" "$mem_limit" "$(to_rel "$json_path")" "$(to_rel "$log_path")" "$notes" \
    >> "$SUMMARY_TSV"
}

make_request() {
  local ctx="$1" task="$2" req_json="$3"
  CTX_SIZE="$ctx" TASK_NAME="$task" MAX_TOKENS="$MAX_TOKENS" LONG_CONTEXT_FILL_RATIO="$LONG_CONTEXT_FILL_RATIO" python3 - <<'PY' > "$req_json"
import json
import os

ctx = int(os.environ["CTX_SIZE"])
task = os.environ["TASK_NAME"]
max_tokens = int(os.environ["MAX_TOKENS"])
ratio = float(os.environ["LONG_CONTEXT_FILL_RATIO"])

target_chars = max(2000, int(ctx * 3.2 * ratio))
source_chunk = """
// BenchmarkContext.cs
public sealed class BenchmarkContext
{
    public required string Repository { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<string> Constraints { get; init; }
    public double Score(double[] values) => values.Where(double.IsFinite).DefaultIfEmpty(0).Average();
}

Operational note: all experiments run through a memory-safe Docker harness with one heavy model server at a time, fixed model paths, reproducible logs, and watchdog timeouts.
"""
filler = (source_chunk * ((target_chars // len(source_chunk)) + 2))[:target_chars]

prompts = {
    "coding_api": (
        "Using the context below, design a small C# API for a resumable local model benchmark runner. "
        "Return concise production-quality code sketches and explain concurrency tradeoffs."
    ),
    "coding_review": (
        "Review the context below as a senior engineer. Identify correctness risks, stability risks, "
        "and missing tests. Then propose concrete fixes."
    ),
    "summary_report": (
        "Summarize the technical report below into 8 bullets, then give 5 action items with owners."
    ),
    "summary_plan": (
        "Compress the long notes below into a durable project memory: decisions, constraints, open risks, "
        "and next steps. Keep it useful for a later coding agent."
    ),
}
instruction = prompts[task]
payload = {
    "model": "local-gguf",
    "messages": [
        {"role": "system", "content": "You are concise and practical. Return only the final answer."},
        {"role": "user", "content": f"/no_think {instruction}\n\n--- LONG CONTEXT START ---\n{filler}\n--- LONG CONTEXT END ---"},
    ],
    "temperature": 0.25 if task.startswith("coding") else 0.35,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": max_tokens,
}
print(json.dumps(payload))
PY
}

measure_response() {
  local req_json="$1" resp_json="$2" out_metrics="$3"
  python3 - "$req_json" "$resp_json" > "$out_metrics" <<'PY'
import json
import pathlib
import sys

req = json.loads(pathlib.Path(sys.argv[1]).read_text())
resp = json.loads(pathlib.Path(sys.argv[2]).read_text())
if resp.get("error"):
    raise SystemExit(f"response error: {resp['error']}")
choices = resp.get("choices") or []
if not choices:
    raise SystemExit("missing choices")
msg = choices[0].get("message", {})
content = str(msg.get("content") or msg.get("reasoning_content") or "")
if not content.strip():
    raise SystemExit("empty response")
prompt_chars = sum(len(str(m.get("content", ""))) for m in req.get("messages", []))
usage = resp.get("usage") or {}
metrics = {
    "prompt_chars": prompt_chars,
    "completion_chars": len(content),
    "prompt_tokens": usage.get("prompt_tokens", ""),
    "completion_tokens": usage.get("completion_tokens", ""),
    "total_tokens": usage.get("total_tokens", ""),
}
print(json.dumps(metrics))
PY
}

run_model_ctx() {
  local alias="$1" model_path="$2" port="$3" gpu_layers="$4" mem_limit="$5" memory_swap="$6" mem_reservation="$7" use_dri="$8" ctx="$9"
  shift 9
  local tasks=("$@")

  local container="llama-text-bench-${port}"
  local server_log="$OUT_DIR/${alias}_ctx${ctx}_server.log"
  local watchdog_log="$OUT_DIR/${alias}_ctx${ctx}_watchdog.log"

  if all_tasks_recorded "$alias" "$ctx" "${tasks[@]}"; then
    echo "skip ${alias} ctx=${ctx} (all requested tasks already recorded)"
    return 0
  fi

  docker rm -f "$container" >/dev/null 2>&1 || true

  set +e
  RUN_WITH_WATCHDOG=1 \
  RUN_MAX_SECONDS="$RUN_MAX_SECONDS" \
  RUN_IDLE_SECONDS="$RUN_IDLE_SECONDS" \
  RUN_POLL_SECONDS="$RUN_POLL_SECONDS" \
  RUN_LOG_PATH="$watchdog_log" \
    "$REPO_ROOT/scripts/run_memsafe.sh" \
      env MODEL_PATH="$model_path" PORT="$port" CTX_SIZE="$ctx" MAX_TOKENS=16 PROMPT_MODE=text THREADS="$THREADS" \
        GPU_LAYERS="$gpu_layers" USE_DRI="$use_dri" EXTRA_ARGS="--jinja --reasoning-budget 0 --reasoning-format none --no-context-shift" \
        MEM_LIMIT="$mem_limit" MEMORY_SWAP="$memory_swap" MEM_RESERVATION="$mem_reservation" OOM_SCORE_ADJ="$OOM_SCORE_ADJ" \
        OUT_JSON="$OUT_JSON_DIR/${alias}_ctx${ctx}_warmup.json" CONTAINER="$container" READY_ATTEMPTS="$READY_ATTEMPTS" CURL_MAX_TIME=300 \
      bash "$REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh" \
      > "$server_log" 2>&1
  local warm_status=$?
  set -e

  if [ "$warm_status" -ne 0 ]; then
    for task in "${tasks[@]}"; do
      append_row "$alias" "$ctx" "$task" "fail-load(${warm_status})" "0" "0" "0" "" "" "" "$port" "$gpu_layers" "$mem_limit" "$OUT_JSON_DIR/${alias}_ctx${ctx}_${task}.json" "$server_log" "server warmup failed"
    done
    return 1
  fi

  # probe_ctx_once stops the container at the end; start a persistent server for per-task timing.
  docker rm -f "$container" >/dev/null 2>&1 || true
  DOCKER_DEVICE_ARGS=()
  if [ "$use_dri" = "1" ]; then
    DOCKER_DEVICE_ARGS+=(--device=/dev/dri)
  fi
  docker run -d --name "$container" \
    --memory="$mem_limit" --memory-swap="$memory_swap" --memory-reservation="$mem_reservation" \
    --oom-score-adj="$OOM_SCORE_ADJ" "${DOCKER_DEVICE_ARGS[@]}" --security-opt label=disable \
    --ipc=host --network=host -v "$HF_ROOT:$HF_ROOT" \
    -e MODEL="$model_path" -e PORT="$port" -e CTX_SIZE="$ctx" -e GPU_LAYERS="$gpu_layers" -e THREADS="$THREADS" \
    -e EXTRA_ARGS="--jinja --reasoning-budget 0 --reasoning-format none --no-context-shift" \
    "$LLAMA_IMAGE" >> "$server_log" 2>&1
  trap 'docker rm -f "$container" >/dev/null 2>&1 || true' RETURN

  local code=""
  for _ in $(seq 1 "$READY_ATTEMPTS"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/v1/models" || true)"
    [ "$code" = "200" ] && break
    sleep "$READY_SLEEP_SECS"
  done
  if [ "$code" != "200" ]; then
    docker logs --tail 200 "$container" >> "$server_log" 2>&1 || true
    for task in "${tasks[@]}"; do
      append_row "$alias" "$ctx" "$task" "fail-ready" "0" "0" "0" "" "" "" "$port" "$gpu_layers" "$mem_limit" "$OUT_JSON_DIR/${alias}_ctx${ctx}_${task}.json" "$server_log" "server not ready"
    done
    return 1
  fi

  for task in "${tasks[@]}"; do
    if [ -n "$(recorded_status "$alias" "$ctx" "$task")" ]; then
      echo "skip ${alias} ctx=${ctx} task=${task} (already recorded)"
      continue
    fi

    local req_json resp_json metrics_json log_path out_json start end duration status
    req_json="$(mktemp "${TMPDIR:-/tmp}/textbench_req.XXXXXX.json")"
    resp_json="$(mktemp "${TMPDIR:-/tmp}/textbench_resp.XXXXXX.json")"
    metrics_json="$(mktemp "${TMPDIR:-/tmp}/textbench_metrics.XXXXXX.json")"
    out_json="$OUT_JSON_DIR/${alias}_ctx${ctx}_${task}.json"
    log_path="$OUT_DIR/${alias}_ctx${ctx}_${task}.log"
    make_request "$ctx" "$task" "$req_json"
    start="$(date +%s)"
    set +e
    curl -sS --connect-timeout 10 --max-time "$CURL_MAX_TIME" \
      "http://127.0.0.1:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d @"$req_json" > "$resp_json" 2> "$log_path"
    status=$?
    set -e
    end="$(date +%s)"
    duration=$((end - start))

    if [ "$status" -eq 0 ] && measure_response "$req_json" "$resp_json" "$metrics_json" 2>> "$log_path"; then
      cp "$resp_json" "$out_json"
      local prompt_chars completion_chars prompt_tokens completion_tokens total_tokens
      prompt_chars="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt_chars"])' "$metrics_json")"
      completion_chars="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["completion_chars"])' "$metrics_json")"
      prompt_tokens="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prompt_tokens",""))' "$metrics_json")"
      completion_tokens="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completion_tokens",""))' "$metrics_json")"
      total_tokens="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("total_tokens",""))' "$metrics_json")"
      append_row "$alias" "$ctx" "$task" "ok" "$duration" "$prompt_chars" "$completion_chars" "$prompt_tokens" "$completion_tokens" "$total_tokens" "$port" "$gpu_layers" "$mem_limit" "$out_json" "$log_path" "completed"
      echo "ok ${alias} ctx=${ctx} task=${task} duration=${duration}s"
    else
      append_row "$alias" "$ctx" "$task" "fail(${status})" "$duration" "0" "0" "" "" "" "$port" "$gpu_layers" "$mem_limit" "$out_json" "$log_path" "request failed"
      echo "fail ${alias} ctx=${ctx} task=${task} status=${status}"
    fi
    rm -f "$req_json" "$resp_json" "$metrics_json"
  done

  docker rm -f "$container" >/dev/null 2>&1 || true
  trap - RETURN
}

IFS=',' read -r -a CONTEXTS <<< "$CONTEXTS_CSV"
IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

MODELS=(
  "qwen36_27b_q4km|$MODEL_ROOT/qwen3.6-27b-gguf/Qwen-Qwen3.6-27B-Q4_K_M.gguf|8162|999|75g|75g|67g|1"
  "qwen35_9b_q8|$MODEL_ROOT/qwen3.5-9b-gguf/Qwen3.5-9B-Q8_0.gguf|8163|999|75g|75g|67g|1"
  "qwen35_27b_q8|$MODEL_ROOT/qwen3.5-27b-gguf/Qwen3.5-27B-Q8_0.gguf|8164|999|75g|75g|67g|1"
  "qwen35_122b_a10b_q4km|$MODEL_ROOT/qwen3.5-122b-a10b-gguf/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf|8165|0|85g|85g|76g|0"
  "gemma4_26b_a4b_q4km|$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf|8166|999|75g|75g|67g|1"
  "qwen3_next_80b_a3b_q5km|$MODEL_ROOT/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf|8167|999|75g|75g|67g|1"
  "qwen3_coder_next_q5km|$MODEL_ROOT/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf|8168|999|75g|75g|67g|1"
  "qwen25_coder_32b_q4km_cpu|$MODEL_ROOT/qwen2.5-coder-32b-instruct-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf|8169|0|75g|75g|67g|0"
  "gpt_oss_120b_mxfp4|$MODEL_ROOT/gpt-oss-120b-gguf/gpt-oss-120b-mxfp4-00001-of-00003.gguf|8170|999|75g|75g|67g|1"
)

for cfg in "${MODELS[@]}"; do
  IFS='|' read -r alias model_path port gpu_layers mem_limit memory_swap mem_reservation use_dri <<< "$cfg"
  if [ -n "$MODEL_FILTER" ] && [[ "$alias" != *"$MODEL_FILTER"* ]]; then
    continue
  fi
  if [ ! -f "$model_path" ]; then
    echo "skip missing model: $alias -> $model_path"
    continue
  fi
  for ctx in "${CONTEXTS[@]}"; do
    run_model_ctx "$alias" "$model_path" "$port" "$gpu_layers" "$mem_limit" "$memory_swap" "$mem_reservation" "$use_dri" "$ctx" "${TASKS[@]}" || true
  done
done

echo "Done. Summary: $(to_rel "$SUMMARY_TSV")"
