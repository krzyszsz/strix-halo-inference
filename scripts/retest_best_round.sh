#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%F)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/retest_${STAMP}_best}"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

mkdir -p "$OUT_DIR"

cat > "$SUMMARY_TSV" <<'EOF'
model	category	workload	status	duration_s	log_path	artifact_path	notes
EOF

LAST_STATUS=0

RUN_MAX_SECONDS_DEFAULT="${RUN_MAX_SECONDS_DEFAULT:-3600}"
# Some validated workloads (1024px / SD3.5-large / upscaling) legitimately run >15 minutes without producing logs.
RUN_IDLE_SECONDS_DEFAULT="${RUN_IDLE_SECONDS_DEFAULT:-900}"
RUN_IDLE_SECONDS_SLOW="${RUN_IDLE_SECONDS_SLOW:-1800}"

run_case() {
  local model="$1"
  local category="$2"
  local workload="$3"
  local artifact="$4"
  local notes="$5"
  local name="$6"
  shift 6
  local cmd="$*"
  local log_path="$OUT_DIR/${name}.log"

  local run_idle="$RUN_IDLE_SECONDS_DEFAULT"
  if [[ "$workload" == *"1024x1024"* ]] || [[ "$model" == "SD3.5-Large" ]] || [[ "$category" == "upscaler" ]]; then
    run_idle="$RUN_IDLE_SECONDS_SLOW"
  fi

  local start_ts end_ts duration
  start_ts="$(date +%s)"
  set +e
  RUN_LOG_PATH="$log_path" \
  RUN_MAX_SECONDS="$RUN_MAX_SECONDS_DEFAULT" \
  RUN_IDLE_SECONDS="$run_idle" \
    "$SCRIPT_DIR/run_memsafe.sh" bash -lc "$cmd"
  LAST_STATUS=$?
  set -e
  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model" "$category" "$workload" "$LAST_STATUS" "$duration" "$log_path" "$artifact" "$notes" \
    >> "$SUMMARY_TSV"

  echo "[${name}] status=${LAST_STATUS} duration=${duration}s"
}

run_case \
  "Qwen-Image" "text-to-image" "1024x1024, steps=30, bf16" \
  "$REPO_ROOT/qwen-image/out/qwen_image_1024_best_retest.png" \
  "best stable profile" \
  "qwen_image_1024_best" \
  "timeout 3600 env WIDTH=1024 HEIGHT=1024 STEPS=30 CURL_MAX_TIME=2400 OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_1024_best_retest.png bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh"

run_case \
  "Qwen-Image-Edit" "image-to-image" "512x512, steps=8, bf16, strength=0.6" \
  "$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_512_best_retest.png" \
  "attempt higher resolution profile" \
  "qwen_image_edit_512_best_attempt" \
  "timeout 3600 env INPUT_IMAGE=$REPO_ROOT/stable-diffusion/out/sd35_sample.png WIDTH=512 HEIGHT=512 STEPS=8 CURL_MAX_TIME=2400 OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_512_best_retest.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit.sh"

if [ "$LAST_STATUS" -ne 0 ]; then
  run_case \
    "Qwen-Image-Edit" "image-to-image" "256x256, steps=8, bf16, strength=0.6" \
    "$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_256_best_retest.png" \
    "fallback stable profile" \
    "qwen_image_edit_256_fallback" \
    "timeout 2400 env INPUT_IMAGE=$REPO_ROOT/stable-diffusion/out/sd35_large_sample.png WIDTH=256 HEIGHT=256 STEPS=8 CURL_MAX_TIME=1800 OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_256_best_retest.png bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit.sh"
fi

run_case \
  "SD3.5-Medium" "text-to-image" "512x512, steps=40, guidance=4.5" \
  "$REPO_ROOT/stable-diffusion/out/sd35_sample_best_retest.png" \
  "default best sample script profile" \
  "sd35_medium_512_best" \
  "timeout 2400 env OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_sample_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_sample.sh"

run_case \
  "SD3.5-Large" "text-to-image" "512x512, steps=20, guidance=3.5" \
  "$REPO_ROOT/stable-diffusion/out/sd35_large_512_best_attempt.png" \
  "attempt higher resolution profile" \
  "sd35_large_512_best_attempt" \
  "timeout 3600 env WIDTH=512 HEIGHT=512 STEPS=20 GUIDANCE=3.5 CURL_MAX_TIME=2400 OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_large_512_best_attempt.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_large_sample.sh"

if [ "$LAST_STATUS" -ne 0 ]; then
  run_case \
    "SD3.5-Large" "text-to-image" "256x256, steps=28, guidance=3.5" \
    "$REPO_ROOT/stable-diffusion/out/sd35_large_256_best_retest.png" \
    "fallback stable profile" \
    "sd35_large_256_fallback" \
    "timeout 2400 env WIDTH=256 HEIGHT=256 STEPS=28 GUIDANCE=3.5 CURL_MAX_TIME=1800 OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_large_256_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_large_sample.sh"
fi

run_case \
  "SDXL-Base" "text-to-image" "512x512, steps=50, guidance=5.0" \
  "$REPO_ROOT/stable-diffusion/out/sdxl_base_best_retest.png" \
  "default best sample script profile" \
  "sdxl_base_512_best" \
  "timeout 2400 env OUT_PATH=$REPO_ROOT/stable-diffusion/out/sdxl_base_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sdxl_base_sample.sh"

run_case \
  "Playground-v2.5" "text-to-image" "1024x1024, steps=20, guidance=3.0, fp16, vae_tiling=1" \
  "$REPO_ROOT/stable-diffusion/out/playground_v25_1024_best_retest.png" \
  "SDXL fine-tune (1024px-native) validation run" \
  "playground_v25_1024_best" \
  "timeout 3600 env DTYPE=float16 VAE_TILING=1 VAE_SLICING=1 RESP_DIR=$OUT_DIR OUT_PATH=$REPO_ROOT/stable-diffusion/out/playground_v25_1024_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_playground_v25_sample.sh"

run_case \
  "Flux2-Klein-4B" "text-to-image" "512x512, steps=4, guidance=1.0" \
  "$REPO_ROOT/stable-diffusion/out/flux2_klein_best_retest.png" \
  "default best sample script profile" \
  "flux2_klein_512_best" \
  "timeout 1800 env OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_sample.sh"

run_case \
  "SD-x4-Upscaler" "upscaler" "512->2048, steps=8, guidance=6.0, noise=10, fp32" \
  "$REPO_ROOT/stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png" \
  "target 2048 upscaling attempt" \
  "sd_x4_upscale_512_to_2048_attempt" \
  "timeout 3600 env INPUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_full.png INPUT_SIDE=512 STEPS=8 NOISE_LEVEL=10 GUIDANCE=6.0 DTYPE=float32 DEVICE=cuda DISABLE_SDP=1 AOTRITON_EXPERIMENTAL=0 OUT_PATH=$REPO_ROOT/stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd_x4_upscale_qwen_image.sh"

if [ "$LAST_STATUS" -ne 0 ]; then
  run_case \
    "SD-x4-Upscaler" "upscaler" "256->1024, steps=10, guidance=6.0, noise=10, fp32" \
    "$REPO_ROOT/stable-diffusion/out/qwen_image_upscaled_1024_best_retest.png" \
    "fallback stable profile" \
    "sd_x4_upscale_256_to_1024_fallback" \
    "timeout 3600 env INPUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_full.png INPUT_SIDE=256 STEPS=10 NOISE_LEVEL=10 GUIDANCE=6.0 DTYPE=float32 DEVICE=cuda DISABLE_SDP=1 AOTRITON_EXPERIMENTAL=0 OUT_PATH=$REPO_ROOT/stable-diffusion/out/qwen_image_upscaled_1024_best_retest.png bash $REPO_ROOT/stable-diffusion/scripts/test_sd_x4_upscale_qwen_image.sh"
fi

run_case \
  "Qwen2.5-VL-7B" "image-to-text" "image=512x512, max_new_tokens=256, cpu-fallback" \
  "$REPO_ROOT/qwen-vl/out/qwen_vl_describe_best_retest.txt" \
  "best stable path on this host (CPU)" \
  "qwen_vl_7b_best_cpu" \
  "timeout 2400 env FORCE_CPU=1 INPUT_IMAGE=$REPO_ROOT/stable-diffusion/out/sd35_sample.png MAX_NEW_TOKENS=256 CURL_MAX_TIME=1200 bash $REPO_ROOT/qwen-vl/scripts/test_qwen_vl_7b.sh && cp -f $REPO_ROOT/qwen-vl/out/qwen_vl_describe.txt $REPO_ROOT/qwen-vl/out/qwen_vl_describe_best_retest.txt"

run_case \
  "Qwen3-Next-80B-A3B" "llm-chat" "ctx=32768, max_tokens=256" \
  "$REPO_ROOT/llama-cpp-vulkan/out/qwen3_next_80b_q5_best_retest.json" \
  "high-context validation" \
  "qwen3_next_80b_best" \
  "timeout 2400 env CTX_SIZE=32768 OUT_DIR=$REPO_ROOT/llama-cpp-vulkan/out bash $REPO_ROOT/llama-cpp-vulkan/scripts/test_qwen3_next_80b.sh && cp -f $REPO_ROOT/llama-cpp-vulkan/out/qwen3_next_80b_q5.json $REPO_ROOT/llama-cpp-vulkan/out/qwen3_next_80b_q5_best_retest.json"

run_case \
  "Qwen3-Coder-Next-Q5" "llm-coding" "ctx=4096, max_tokens=256" \
  "$REPO_ROOT/llama-cpp-vulkan/out/qwen3_coder_next_q5_best_retest.json" \
  "default stable coding profile" \
  "qwen3_coder_next_best" \
  "timeout 2400 env CTX_SIZE=4096 OUT_DIR=$REPO_ROOT/llama-cpp-vulkan/out bash $REPO_ROOT/llama-cpp-vulkan/scripts/test_qwen3_coder_next_q5.sh && cp -f $REPO_ROOT/llama-cpp-vulkan/out/qwen3_coder_next_q5.json $REPO_ROOT/llama-cpp-vulkan/out/qwen3_coder_next_q5_best_retest.json"

run_case \
  "Qwen2.5-Coder-32B" "llm-coding" "ctx=2048, max_tokens=256" \
  "$REPO_ROOT/llama-cpp-vulkan/out/qwen25_coder_32b_best_retest.json" \
  "default stable coding fallback profile" \
  "qwen25_coder_32b_best" \
  "timeout 2400 env CTX_SIZE=2048 OUT_DIR=$REPO_ROOT/llama-cpp-vulkan/out bash $REPO_ROOT/llama-cpp-vulkan/scripts/test_qwen25_coder_32b.sh && cp -f $REPO_ROOT/llama-cpp-vulkan/out/qwen25_coder_32b.json $REPO_ROOT/llama-cpp-vulkan/out/qwen25_coder_32b_best_retest.json"

run_case \
  "GPT-OSS-120B-MXFP4" "llm-chat" "ctx=2048, max_tokens=256" \
  "$REPO_ROOT/llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_best_retest.json" \
  "validated under 75g memory cap" \
  "gpt_oss_120b_best" \
  "timeout 3600 env OUT_PATH=$REPO_ROOT/llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_best_retest.json bash $REPO_ROOT/llama-cpp-vulkan/scripts/test_gpt_oss_120b_mxfp4.sh"

run_case \
  "Kokoro-82M" "audio-tts" "transcript~158s output, voice=af_heart, speed=1.0" \
  "$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav" \
  "long-form tts validation" \
  "audio_kokoro_best" \
  "timeout 2400 env OUT_WAV=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav OUT_SUMMARY=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_summary.json bash $REPO_ROOT/audio/scripts/test_kokoro_podcast.sh"

run_case \
  "faster-whisper-small" "audio-stt" "input~158s, model=small, cpu=int8, srt output" \
  "$REPO_ROOT/audio/out/podcast_kokoro_best_retest.srt" \
  "subtitle extraction validation" \
  "audio_stt_best" \
  "timeout 2400 env INPUT_AUDIO=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav OUT_SRT=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.srt OUT_TXT=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_transcript.txt OUT_SUMMARY=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_stt_summary.json bash $REPO_ROOT/audio/scripts/test_faster_whisper_subtitles.sh"

echo "Summary: $SUMMARY_TSV"
