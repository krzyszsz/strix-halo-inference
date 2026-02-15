#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

STAMP="$(date -u +%F)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/publish}"
SUMMARY_TSV="$OUT_DIR/summary_final.tsv"
RESUME="${RESUME:-0}"

# This script is intended for publish-day reruns: keep outputs fresh and avoid
# accidentally mixing evidence across days.
if [ "${CLEAN_OUT_DIR:-1}" = "1" ]; then
  if [ "$RESUME" = "1" ]; then
    echo "RESUME=1 set: skipping OUT_DIR cleanup (set CLEAN_OUT_DIR=0 to silence this message)." >&2
  else
  case "$OUT_DIR" in
    "$REPO_ROOT"/reports/*)
      rm -rf "$OUT_DIR"
      ;;
    *)
      echo "Refusing to clean OUT_DIR outside repo reports/: $OUT_DIR" >&2
      ;;
  esac
  fi
fi

mkdir -p "$OUT_DIR"

if [ "$RESUME" != "1" ] || [ ! -s "$SUMMARY_TSV" ]; then
  cat >"$SUMMARY_TSV" <<'EOF'
name	category	workload	status	duration_s	log_path	artifacts	notes
EOF
fi

RUN_MAX_SECONDS_DEFAULT="${RUN_MAX_SECONDS_DEFAULT:-3600}"
RUN_IDLE_SECONDS_DEFAULT="${RUN_IDLE_SECONDS_DEFAULT:-600}"
RUN_IDLE_SECONDS_SLOW="${RUN_IDLE_SECONDS_SLOW:-1800}"

LAST_STATUS=0

case_done() {
  local name="$1"
  [ -s "$SUMMARY_TSV" ] || return 1
  # Header: name	category	...
  awk -F'\t' -v n="$name" 'NR>1 && $1==n && $4=="0" {found=1} END{exit found?0:1}' "$SUMMARY_TSV"
}

run_case() {
  local name="$1"
  local category="$2"
  local workload="$3"
  local artifacts="$4"
  local notes="$5"
  local idle_seconds="$6"
  local max_seconds="$7"
  shift 7
  local cmd="$*"
  local log_path="$OUT_DIR/${name}.log"
  local log_path_rel
  log_path_rel="$(realpath --relative-to="$REPO_ROOT" "$log_path" 2>/dev/null || echo "$log_path")"

  if [ "$RESUME" = "1" ] && case_done "$name"; then
    echo "[${name}] skip (already status=0 in $SUMMARY_TSV)"
    return 0
  fi

  # Ensure logs are per-attempt and don't silently accumulate across aborted runs.
  : >"$log_path"

  local start_ts end_ts duration
  start_ts="$(date +%s)"

  set +e
  RUN_LOG_PATH="$log_path" \
  RUN_MAX_SECONDS="${max_seconds:-$RUN_MAX_SECONDS_DEFAULT}" \
  RUN_IDLE_SECONDS="${idle_seconds:-$RUN_IDLE_SECONDS_DEFAULT}" \
    "$REPO_ROOT/scripts/run_memsafe.sh" bash -lc "set -euo pipefail; $cmd"
  LAST_STATUS=$?
  set -e

  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$category" "$workload" "$LAST_STATUS" "$duration" "$log_path_rel" "$artifacts" "$notes" \
    >>"$SUMMARY_TSV"

  echo "[${name}] status=${LAST_STATUS} duration=${duration}s log=${log_path}"
}

# -----------------
# Text to Image
# -----------------

run_case \
  "qwen_image_2512_512" "text-to-image" "512x512 steps=20 bf16" \
  "qwen-image/out/qwen_image_2512_512_2026-02-11.png" \
  "Qwen-Image-2512 validated profile" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/qwen-image-2512 WIDTH=512 HEIGHT=512 STEPS=20 DTYPE=bfloat16 OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_2512_512_2026-02-11.png bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh"

run_case \
  "qwen_image_512" "text-to-image" "512x512 steps=30 bf16" \
  "qwen-image/out/qwen_image_512_75g_retest2.png" \
  "Qwen-Image 512px run" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/qwen-image WIDTH=512 HEIGHT=512 STEPS=30 DTYPE=bfloat16 OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh"

run_case \
  "qwen_image_1024" "text-to-image" "1024x1024 steps=30 bf16" \
  "qwen-image/out/qwen_image_1024_75g_retest2.png" \
  "Qwen-Image 1024px run" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/qwen-image WIDTH=1024 HEIGHT=1024 STEPS=30 DTYPE=bfloat16 CURL_MAX_TIME=2400 OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_1024_75g_retest2.png bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh"

run_case \
  "flux2_klein_base_4b_512" "text-to-image" "512x512 steps=4 guidance=1.0 fp32" \
  "stable-diffusion/out/flux2_klein_base_4b_512_2026-02-11.png" \
  "Flux2 klein-base 4B" \
  "$RUN_IDLE_SECONDS_DEFAULT" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/flux2-klein-base-4b DTYPE=float32 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=1.0 OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_base_4b_512_2026-02-11.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_sample.sh"

run_case \
  "flux2_klein_4b_512" "text-to-image" "512x512 steps=4 guidance=1.0 fp32" \
  "stable-diffusion/out/flux2_klein_best_retest.png" \
  "Flux2 klein 4B" \
  "$RUN_IDLE_SECONDS_DEFAULT" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/flux2-klein-4b DTYPE=float32 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=1.0 OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_sample.sh"

run_case \
  "flux2_klein_9b_512_t2i" "text-to-image" "512x512 steps=4 guidance=1.0 bf16 cpu-offload" \
  "stable-diffusion/out/flux2_klein_9b_512_t2i_reconfirm_2026-02-12.png" \
  "Flux2 klein 9B probe" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/flux2-klein-9b HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=1.0 MODEL_CPU_OFFLOAD=1 OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_9b_512_t2i_reconfirm_2026-02-12.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_probe.sh"

run_case \
  "flux2_dev_bnb4_512_t2i" "text-to-image" "512x512 steps=4 guidance=3.0 bf16 cpu-offload" \
  "stable-diffusion/out/flux2_dev_bnb4_512_t2i_reconfirm_2026-02-12.png" \
  "diffusers/FLUX.2-dev-bnb-4bit (t2i)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=3.0 DTYPE=bfloat16 MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_t2i_reconfirm_2026-02-12.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh"

run_case \
  "flux2_dev_bnb4_512_i2i" "image-to-image" "512x512 steps=4 guidance=3.0 bf16 cpu-offload" \
  "stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png" \
  "diffusers/FLUX.2-dev-bnb-4bit (i2i)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=3.0 DTYPE=bfloat16 MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 INIT_IMAGE=qwen-image/out/qwen_image_512_75g_retest2.png PROMPT='convert this into a cinematic oil painting while preserving composition' OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh"

run_case \
  "flux2_dev_bnb4_512_multi" "multi-image" "512x512 steps=4 guidance=3.0 bf16 cpu-offload" \
  "stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png" \
  "diffusers/FLUX.2-dev-bnb-4bit (multi)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=3.0 DTYPE=bfloat16 MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 INIT_IMAGE=qwen-image/out/qwen_image_512_75g_retest2.png,qwen-image/out/qwen_image_2512_512_2026-02-11.png PROMPT='blend both reference images into one coherent cinematic scene with realistic lighting' OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh"

run_case \
  "flux2_dev_nvfp4_expected_fail" "text-to-image" "512x512 steps=4 (expected fail)" \
  "n/a" \
  "Expected to fail in this stack; keep for regression tracking" \
  "$RUN_IDLE_SECONDS_DEFAULT" 900 \
  "set +e; env MODEL_ID=$MODEL_ROOT/flux2-dev-nvfp4 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=3.0 DTYPE=bfloat16 MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_nvfp4_expected_fail.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh; exit 0"

run_case \
  "sd35_medium_512" "text-to-image" "512x512 steps=40 guidance=4.5" \
  "stable-diffusion/out/sd35_sample_best_retest.png" \
  "SD3.5 medium" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_sample_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_sample.sh"

run_case \
  "sd35_large_512" "text-to-image" "512x512 steps=20 guidance=3.5" \
  "stable-diffusion/out/sd35_large_512_best_attempt.png" \
  "SD3.5 large" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env WIDTH=512 HEIGHT=512 STEPS=20 GUIDANCE=3.5 CURL_MAX_TIME=2400 OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_large_512_best_attempt.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_large_sample.sh"

run_case \
  "sdxl_base_512" "text-to-image" "512x512 steps=50 guidance=5.0" \
  "stable-diffusion/out/sdxl_base_best_retest.png" \
  "SDXL base" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env OUT_PATH=$REPO_ROOT/stable-diffusion/out/sdxl_base_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sdxl_base_sample.sh"

run_case \
  "playground_v25_1024" "text-to-image" "1024x1024 steps=20 guidance=3.0 fp16" \
  "stable-diffusion/out/playground_v25_1024_2026-02-12.png" \
  "Playground v2.5 SDXL fine-tune" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env OUT_PATH=$REPO_ROOT/stable-diffusion/out/playground_v25_1024_2026-02-12.png bash $REPO_ROOT/stable-diffusion/scripts/test_playground_v25_sample.sh"

# -----------------
# Image to Image (Qwen-Image-Edit family)
# -----------------

run_case \
  "qwen_image_edit_base_256_compat" "image-to-image" "256x256 steps=4 strength=0.6 bf16" \
  "qwen-image-edit/out/qwen_image_edit_single_compat_2026-02-11.png" \
  "Single-image API compatibility check" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODEL_ID=$MODEL_ROOT/qwen-image-edit INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png HEIGHT=256 WIDTH=256 STEPS=4 STRENGTH=0.6 DTYPE=bfloat16 OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_single_compat_2026-02-11.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit.sh"

run_case \
  "qwen_image_edit_2509_single_512" "image-to-image" "512x512 steps=4 bf16 seq-offload swap=140g" \
  "qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png" \
  "Qwen-Image-Edit-2509 plus (single)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODE=single MODEL_ID=$MODEL_ROOT/qwen-image-edit-2509 INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png HEIGHT=512 WIDTH=512 STEPS=4 MEMORY_SWAP=140g OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh"

run_case \
  "qwen_image_edit_2509_multi_512" "multi-image" "2 inputs 512x512 steps=8 bf16 seq-offload swap=140g" \
  "qwen-image-edit/out/qwen_image_edit_2509_multi_512_human_insert_steps8_75g_swap140.png" \
  "Qwen-Image-Edit-2509 plus (multi)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODE=multi MODEL_ID=$MODEL_ROOT/qwen-image-edit-2509 HEIGHT=512 WIDTH=512 STEPS=8 MEMORY_SWAP=140g INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_single_compat_2026-02-11.png INPUT_IMAGE_B=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png PROMPT='Create one coherent scene by placing the human from image A into image B. Keep the person identity and face natural, match lighting and perspective, keep both subjects visible.' OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2509_multi_512_human_insert_steps8_75g_swap140.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh"

run_case \
  "qwen_image_edit_2511_single_512" "image-to-image" "512x512 steps=4 bf16 seq-offload swap=140g" \
  "qwen-image-edit/out/qwen_image_edit_2511_single_512_seqoffload_bf16_75g_swap140_test.png" \
  "Qwen-Image-Edit-2511 plus (single)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODE=single MODEL_ID=$MODEL_ROOT/qwen-image-edit-2511 INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png HEIGHT=512 WIDTH=512 STEPS=4 MEMORY_SWAP=140g OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2511_single_512_seqoffload_bf16_75g_swap140_test.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh"

run_case \
  "qwen_image_edit_2511_multi_512_move_person" "multi-image" "2 inputs 512x512 steps=12 cfg=2.0 seed=3456 bf16 seq-offload swap=140g" \
  "qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps12_cfg2_seed3456_75g_swap140_2026-02-13.png" \
  "Qwen-Image-Edit-2511 plus (multi move-person)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env MODE=multi MODEL_ID=$MODEL_ROOT/qwen-image-edit-2511 HEIGHT=512 WIDTH=512 STEPS=12 TRUE_CFG_SCALE=2.0 SEED=3456 MEMORY_SWAP=140g INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_a_512_seed1234.png INPUT_IMAGE_B=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png PROMPT='Take the person from image A and insert them into image B. Keep the original person from image B. Place the inserted person on the left side of image B, standing naturally near the other person. Preserve both faces, match lighting and perspective, keep the coffee shop background unchanged, sharp focus.' OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps12_cfg2_seed3456_75g_swap140_2026-02-13.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh"

# -----------------
# Upscaling
# -----------------

run_case \
  "sd_x4_upscale_512_to_2048" "upscaler" "512->2048 steps=8 guidance=6.0 noise=10 fp16 aotriton sdp" \
  "stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png" \
  "SD x4 upscaler" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "env INPUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_1024_75g_retest2.png INPUT_SIDE=512 STEPS=8 NOISE_LEVEL=10 GUIDANCE=6.0 DTYPE=float16 DEVICE=cuda DISABLE_SDP=0 AOTRITON_EXPERIMENTAL=1 OUT_PATH=$REPO_ROOT/stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd_x4_upscale_qwen_image.sh"

# -----------------
# Image to Text
# -----------------

run_case \
  "qwen_vl_7b_cpu" "image-to-text" "cpu max_new_tokens=256" \
  "qwen-vl/out/qwen_vl_describe_75g_retest2.txt" \
  "Qwen2.5-VL CPU fallback" \
  "$RUN_IDLE_SECONDS_DEFAULT" 2400 \
  "env FORCE_CPU=1 MAX_NEW_TOKENS=256 INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png OUT_PATH=$REPO_ROOT/qwen-vl/out/qwen_vl_describe_75g_retest2.txt bash $REPO_ROOT/qwen-vl/scripts/test_qwen_vl_7b.sh"

# -----------------
# Detection / Pose
# -----------------

run_case \
  "yolo26n_detect_bus" "vision-detection" "bus.jpg conf=0.25 imgsz=640 device=cpu" \
  "vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json, vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg" \
  "YOLO26n detect" \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "env MODEL_NAME=yolo26n.pt DEVICE=cpu INPUT_IMAGE=$REPO_ROOT/vision-detection/input/bus.jpg OUT_JSON=$REPO_ROOT/vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json OUT_IMAGE=$REPO_ROOT/vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg bash $REPO_ROOT/vision-detection/scripts/test_yolo_detect.sh"

run_case \
  "yolo26n_pose_bus" "vision-pose" "bus.jpg conf=0.25 imgsz=640 device=cpu" \
  "vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json, vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg" \
  "YOLO26n pose" \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "env MODEL_NAME=yolo26n-pose.pt DEVICE=cpu INPUT_IMAGE=$REPO_ROOT/vision-detection/input/bus.jpg OUT_JSON=$REPO_ROOT/vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json OUT_IMAGE=$REPO_ROOT/vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg bash $REPO_ROOT/vision-detection/scripts/test_yolo_pose.sh"

# -----------------
# Face Recognition
# -----------------

run_case \
  "vision_face_match_demo" "face-recognition" "synthetic 2x2 collage hit-rate>=0.5" \
  "vision-face/out/face_match_annotated.png, vision-face/out/face_match_summary.json" \
  "UltraFace + ArcFace ONNX CPU demo" \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "bash $REPO_ROOT/vision-face/scripts/test_face_match.sh"

# -----------------
# Large-context LLM probes (llama.cpp)
# -----------------

run_case \
  "llama_qwen3_next_ctx196608" "llm-chat" "ctx=196608 max_tokens=1024" \
  "llama-cpp-vulkan/out/qwen3_next_80b_q5_ctx_196608_75g_retest2.json" \
  "Qwen3-Next context probe" \
  "$RUN_IDLE_SECONDS_DEFAULT" 3600 \
  "env MODEL_PATH=$MODEL_ROOT/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf PORT=8113 CTX_SIZE=196608 MAX_TOKENS=1024 PROMPT_MODE=text THREADS=8 GPU_LAYERS=999 OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/qwen3_next_80b_q5_ctx_196608_75g_retest2.json bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh"

run_case \
  "llama_qwen3_coder_next_ctx196608" "llm-coding" "ctx=196608 max_tokens=1536" \
  "llama-cpp-vulkan/out/qwen3_coder_next_q5_ctx_196608_75g_retest2.json" \
  "Qwen3-Coder-Next context probe" \
  "$RUN_IDLE_SECONDS_DEFAULT" 3600 \
  "env MODEL_PATH=$MODEL_ROOT/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf PORT=8114 CTX_SIZE=196608 MAX_TOKENS=1536 PROMPT_MODE=coding THREADS=8 GPU_LAYERS=999 OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/qwen3_coder_next_q5_ctx_196608_75g_retest2.json bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh"

run_case \
  "llama_qwen25_coder_32b_ctx131072" "llm-coding" "ctx=131072 max_tokens=1536" \
  "llama-cpp-vulkan/out/qwen25_coder_32b_q4_ctx_131072_75g_retest2.json" \
  "Qwen2.5-Coder-32B context probe" \
  "$RUN_IDLE_SECONDS_DEFAULT" 3600 \
  "env MODEL_PATH=$MODEL_ROOT/qwen2.5-coder-32b-instruct-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf PORT=8116 CTX_SIZE=131072 MAX_TOKENS=1536 PROMPT_MODE=coding THREADS=8 GPU_LAYERS=999 OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/qwen25_coder_32b_q4_ctx_131072_75g_retest2.json bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh"

run_case \
  "llama_gpt_oss_120b_ctx131072" "llm-chat" "ctx=131072 max_tokens=1024" \
  "llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_ctx_131072_75g_retest2.json" \
  "GPT-OSS-120B MXFP4 context probe" \
  "$RUN_IDLE_SECONDS_DEFAULT" 3600 \
  "env MODEL_PATH=$MODEL_ROOT/gpt-oss-120b-gguf/gpt-oss-120b-mxfp4-00001-of-00003.gguf PORT=8115 CTX_SIZE=131072 MAX_TOKENS=1024 PROMPT_MODE=text THREADS=8 GPU_LAYERS=999 OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_ctx_131072_75g_retest2.json bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh"

# -----------------
# Audio
# -----------------

run_case \
  "audio_kokoro_tts" "audio-tts" "podcast transcript -> wav" \
  "audio/out/podcast_kokoro_best_retest.wav" \
  "Kokoro long-form TTS" \
  "$RUN_IDLE_SECONDS_DEFAULT" 2400 \
  "env OUT_WAV=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav OUT_SUMMARY=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_summary.json bash $REPO_ROOT/audio/scripts/test_kokoro_podcast.sh"

run_case \
  "audio_faster_whisper_subtitles" "audio-stt" "wav -> srt" \
  "audio/out/podcast_kokoro_best_retest.srt" \
  "Subtitle extraction" \
  "$RUN_IDLE_SECONDS_DEFAULT" 2400 \
  "env INPUT_AUDIO=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav OUT_SRT=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.srt OUT_TXT=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_transcript.txt OUT_SUMMARY=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_stt_summary.json bash $REPO_ROOT/audio/scripts/test_faster_whisper_subtitles.sh"

run_case \
  "audio_voxtral_mini_3b_transcribe" "audio-stt" "30s clip -> transcript" \
  "audio/out/voxtral_mini_3b_2507_transcript.txt" \
  "Voxtral 3B fallback (slow but works)" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "bash $REPO_ROOT/audio/scripts/test_voxtral_mini_3b_2507_transcribe.sh"

# -----------------
# 3D Reconstruction
# -----------------

run_case \
  "vggt_reconstruct" "3d-reconstruction" "12 photos -> ply" \
  "reconstruction-3d/out/south_building/south_building_points.ply" \
  "VGGT-1B small demo" \
  "$RUN_IDLE_SECONDS_DEFAULT" 3600 \
  "test -d $REPO_ROOT/reconstruction-3d/data/south_building/images || bash $REPO_ROOT/reconstruction-3d/scripts/download_south_building_views.sh && bash $REPO_ROOT/reconstruction-3d/scripts/run_vggt_reconstruct.sh"

# -----------------
# Video
# -----------------

run_case \
  "wan21_t2v" "video-gen" "672x384 17 frames steps=8" \
  "video/out/wan21_t2v_sample.mp4" \
  "Wan2.1 T2V sample" \
  "$RUN_IDLE_SECONDS_SLOW" "$RUN_MAX_SECONDS_DEFAULT" \
  "bash $REPO_ROOT/video/scripts/test_wan21_t2v_sample.sh"

# -----------------
# LLM tools demos
# -----------------

run_case \
  "llm_quantize_demo" "llm-tools" "fp16 gguf -> q4 + smoke test" \
  "reports/quantize/quantize_outputs.json" \
  "Quantization in container" \
  "$RUN_IDLE_SECONDS_DEFAULT" 3600 \
  "env OUT_DIR=$REPO_ROOT/reports/quantize bash $REPO_ROOT/llm-quantize/scripts/quantize_qwen25_05b_fp16_to_q4km.sh"

run_case \
  "llm_finetune_demo" "llm-tools" "cpu lora sft" \
  "reports/finetune/finetune_comparison.json" \
  "Fine-tuning in container" \
  "$RUN_IDLE_SECONDS_SLOW" 5400 \
  "env OUT_DIR=$REPO_ROOT/reports/finetune bash $REPO_ROOT/llm-finetune/scripts/finetune_smollm2_135m_lora_demo.sh"

# -----------------
# MCP workflows
# -----------------

run_case \
  "mcp_playwright_bbc_scrape" "mcp" "bbc scrape + screenshots (may stop on CAPTCHA)" \
  "mcp/out/bbc_$(date -u +%F)/summary.json" \
  "Network-dependent; expected to stop if blocked" \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "python $REPO_ROOT/mcp/scripts/playwright_bbc_scrape.py"

run_case \
  "mcp_playwright_reliability_demo" "mcp" "bbc reliability demo (no bypass)" \
  "mcp/out/bbc_reliable_$(date -u +%F)/summary.json" \
  "Network-dependent; may stop if blocked" \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "python $REPO_ROOT/mcp/scripts/playwright_reliability_demo.py"

run_case \
  "mcp_excel_calc_demo" "mcp" "excel calc demo" \
  "mcp/out/excel_$(date -u +%F)/summary.json" \
  "Offline" \
  "$RUN_IDLE_SECONDS_DEFAULT" 600 \
  "python $REPO_ROOT/mcp/scripts/excel_calc_demo.py"

run_case \
  "mcp_python_shell_demo" "mcp" "python + shell demo" \
  "mcp/out/python_shell_$(date -u +%F)/summary.json" \
  "Offline" \
  "$RUN_IDLE_SECONDS_DEFAULT" 600 \
  "python $REPO_ROOT/mcp/scripts/python_shell_demo.py"

# -----------------
# Agentic coding demos (non-interactive reruns)
# -----------------

run_case \
  "agentic_dotnet_demo_build_and_run" "agentic" "build + run dotnet minimal api container" \
  "agentic/out/dotnet_run_response.json" \
  "Does not rerun Aider itself; validates the resulting repo still builds/runs." \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "bash $REPO_ROOT/agentic/scripts/build_dotnet_container.sh"

run_case \
  "agentic_ga_optimizer_tests" "agentic" "dotnet test ga-optimizer-demo" \
  "agentic/ga-optimizer-demo/out/tests_post_impl_recheck_publish.log" \
  "Validates the staged GA optimizer solution still passes tests." \
  "$RUN_IDLE_SECONDS_DEFAULT" 1800 \
  "env OUT_LOG=$REPO_ROOT/agentic/ga-optimizer-demo/out/tests_post_impl_recheck_publish.log bash $REPO_ROOT/agentic/ga-optimizer-demo/scripts/run_tests.sh"

echo "Summary written: $SUMMARY_TSV"
