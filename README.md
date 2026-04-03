# AMD Ryzen AI Max+ 395 (Strix Halo) Dockerized Local AI Inference Stack

This repository documents a set of local AI inference tests run on an AMD Ryzen AI Max+ 395 (Strix Halo) mini-PC (`96GB` unified memory). Most workloads are containerized with Docker for isolation, and everything is script-driven so it should be easy to reproduce. I put together this repo mostly "vibe-coding", after struggling with early-driver behavior and spotty ROCm support on this platform.

Even with AI help, it was still a bit of work: hangs, kernel panics, memory deadlocks, and an iterative prodding to get stable settings. This is a best-effort snapshot of one machine and one software stack as of **April 2026**. There are likely mistakes or sub-optimal values.

Model focus: I picked local models primarily for coding / agentic / chat workflows, plus a set of image-generation models. Most of the other entries are there to explore what's possible on this hardware/software stack, so many of them are intentionally small. The largest model tested here is `120B` parameters, so it is not even in a race with current cloud models on raw capability. Cloud models will always be better, but local AI still makes sense for independence, privacy/data safety, and hands-on learning/experimentation.

Covered (working scripts + evidence logs):
- Text-to-image generation (multiple model families).
- Image editing and multi-image composition (insert/move subjects between inputs).
- Diffusion upscaling (up to `2048x2048` output).
- Text-to-video generation.
- Image-to-text captioning (VLM).
- Coding/chat LLM inference with large-context configs.
- MCP workflows (tools + servers).
- Agentic coding experiments (plan-driven, tests-first demos).
- Audio: long-form TTS plus speech-to-text and subtitle extraction.
- Vision: object detection + pose keypoints, and privacy-safe face matching.
- Multi-view 3D reconstruction + browser viewer.

![Model output collage](docs/images/strix-halo-model-collage-2026-03-08-12.png)
Caption: Combined preview of selected outputs generated in this repository (12 tiles: `Qwen-Image-2512`, `Qwen-Image`, `Playground v2.5`, `FLUX.2-klein-9B (20 steps)`, `diffusers/FLUX.2-dev-bnb-4bit (20 steps)`, `Qwen-Image-Edit-2511 (20 steps, multi)`, `SD3.5 Medium`, `SD x4 Upscaler`, `YOLO26 detect`, `VGGT-1B`, `Wan2.1-T2V`, `UltraFace+ArcFace`).

## Contents

- [Quick Find](#quick-find)
- [Tested Models: Base Info and Test Overview](#tested-models-base-info-and-test-overview)
- [Text to Image](#text-to-image)
  - [Qwen-Image-2512 (latest)](#qwen-image-2512-latest)
  - [Qwen-Image (512 and 1024)](#qwen-image-512-and-1024)
  - [FLUX.2-klein-base-4B](#flux2-klein-base-4b)
  - [FLUX.2-klein-4B](#flux2-klein-4b)
  - [FLUX.2-klein-9B](#flux2-klein-9b)
  - [FLUX.2-dev-bnb-4bit (diffusers)](#flux2-dev-bnb-4bit-diffusers)
  - [FLUX.2-dev-NVFP4 (fails in this stack)](#flux2-dev-nvfp4-fails-in-this-stack)
  - [Stable Diffusion 3.5 Medium](#stable-diffusion-35-medium)
  - [Stable Diffusion 3.5 Large](#stable-diffusion-35-large)
  - [SDXL Base](#sdxl-base)
  - [Playground v2.5 (SDXL fine-tune)](#playground-v25-sdxl-fine-tune)
- [Image to Image](#image-to-image)
  - [Qwen-Image-Edit](#qwen-image-edit)
  - [Qwen-Image-Edit-2509](#qwen-image-edit-2509)
  - [Qwen-Image-Edit-2511](#qwen-image-edit-2511)
- [Upscaling](#upscaling)
  - [SD x4 Upscaler](#sd-x4-upscaler)
- [Image to Text](#image-to-text)
  - [Qwen2.5-VL-7B-Instruct](#qwen25-vl-7b-instruct)
  - [Qwen3.5 VLM suite (cross-links)](#qwen35-vlm-suite-cross-links)
  - [Gemma 4 26B-A4B (cross-links)](#gemma-4-26b-a4b-cross-links)
- [Detection and Pose](#detection-and-pose)
  - [YOLO26n (detect)](#yolo26n-detect)
  - [YOLO26n-pose](#yolo26n-pose)
- [Face Recognition](#face-recognition)
  - [UltraFace and ArcFace (ONNX)](#ultraface-and-arcface-onnx)
- [Coding and General Chat](#coding-and-general-chat)
  - [Gemma 4 26B-A4B (GGUF, VLM)](#gemma-4-26b-a4b-gguf-vlm)
  - [Qwen3.5-9B Q8_0 (GGUF, VLM)](#qwen35-9b-q8_0-gguf-vlm)
  - [Qwen3.5-27B Q8_0 (GGUF, VLM)](#qwen35-27b-q8_0-gguf-vlm)
  - [Qwen3.5-122B-A10B Q4_K_M (GGUF, VLM)](#qwen35-122b-a10b-q4_k_m-gguf-vlm)
  - [Qwen3-Next-80B-A3B (GGUF)](#qwen3-next-80b-a3b-gguf)
  - [Qwen3-Coder-Next (GGUF)](#qwen3-coder-next-gguf)
  - [Qwen2.5-Coder-32B (GGUF)](#qwen25-coder-32b-gguf)
  - [GPT-OSS-120B MXFP4 (GGUF)](#gpt-oss-120b-mxfp4-gguf)
  - [VS Code (Continue) integration](#vs-code-continue-integration)
- [MCP](#mcp)
  - [Gemma 4 Playwright + Docker shell tool demo](#gemma-4-playwright--docker-shell-tool-demo)
  - [Playwright BBC scrape](#playwright-bbc-scrape)
  - [Playwright reliability demo](#playwright-reliability-demo)
  - [Excel calc demo](#excel-calc-demo)
  - [Python and shell demo](#python-and-shell-demo)
- [Audio](#audio)
  - [Kokoro-82M (TTS)](#kokoro-82m-tts)
  - [faster-whisper (subtitles)](#faster-whisper-subtitles)
  - [Voxtral-Mini-3B-2507 (fallback)](#voxtral-mini-3b-2507-fallback)
- [3D Reconstruction](#3d-reconstruction)
  - [VGGT-1B](#vggt-1b)
- [Video](#video)
  - [Wan2.1-T2V-1.3B](#wan21-t2v-13b)
- [LLM Quantization and Fine-Tuning Demos](#llm-quantization-and-fine-tuning-demos)
  - [Quantization demo (GGUF)](#quantization-demo-gguf)
  - [Fine-tuning demo (CPU LoRA)](#fine-tuning-demo-cpu-lora)
- [Agentic Coding Experiments](#agentic-coding-experiments)
  - [Aider and local coder model demo](#aider-and-local-coder-model-demo)
  - [Staged GA optimizer demo (with MCP)](#staged-ga-optimizer-demo-with-mcp)
- [Tested Environment (Actual Values)](#tested-environment-actual-values)
- [Host Setup (Fedora 43 / Strix Halo)](#host-setup-fedora-43--strix-halo)
- [Initial Machine Configuration (Fedora 43 + Kernel 6.18)](#initial-machine-configuration-fedora-43--kernel-618)
- [Containers](#containers)
- [Model Downloads](#model-downloads)
- [Cleanup Policy](#cleanup-policy)
- [Passwordless sudo and Non-root copy](#passwordless-sudo-and-non-root-copy)

## Quick Find

- Looking for one-line model capabilities/timings/licenses: go to [Tested Models: Base Info and Test Overview](#tested-models-base-info-and-test-overview).
- Looking for text-to-image generation: go to [Text to Image](#text-to-image).
- Looking for image edits or multi-image composition: go to [Image to Image](#image-to-image).
- Looking for image captioning / visual understanding: start at [Image to Text](#image-to-text), then see [Qwen3.5 VLM suite (cross-links)](#qwen35-vlm-suite-cross-links) and [Gemma 4 26B-A4B (cross-links)](#gemma-4-26b-a4b-cross-links).
- Looking for coding/chat LLMs: go to [Coding and General Chat](#coding-and-general-chat).
- Looking for detection, pose, and face matching: go to [Detection and Pose](#detection-and-pose) and [Face Recognition](#face-recognition).
- Looking for audio, video, and 3D: go to [Audio](#audio), [Video](#video), and [3D Reconstruction](#3d-reconstruction).
- Looking for setup/build/download steps: go to [Containers](#containers), [Model Downloads](#model-downloads), and [Cleanup Policy](#cleanup-policy).
- Looking for Gemma tool-use with MCP (Playwright + dockerized shell): go to [Gemma 4 Playwright + Docker shell tool demo](#gemma-4-playwright--docker-shell-tool-demo).

## Tested Models: Base Info and Test Overview

This table summarizes the publish-day rerun (strict memory limits, scripted and containerized):
- `reports/publish/summary_final.tsv`
Times are approximate wall-clock durations and include pre/post cleanup wrapper overhead.
`Model Size` and `Release date` are best-effort values from model cards/HF API metadata (`createdAt`) and may differ from upstream announcement dates. If a model card doesn't state a parameter count, I compute it from locally downloaded weights (e.g., `.safetensors` header) and include the evidence file in the table.

| Model | Category | Model Size (Params) | Release date | Max Tested Workload (Validated) | Approx Time | Mode | License | Model card | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen-Image-2512 (latest) | text-to-image | `20B` | `2025-12-30` | `512x512`, `steps=20`, `bf16` | `~3m` | GPU | `apache-2.0` | [`Qwen/Qwen-Image-2512`](https://huggingface.co/Qwen/Qwen-Image-2512) | `reports/publish/qwen_image_2512_512.log`, `qwen-image/out/qwen_image_2512_512_2026-02-11.png` |
| Qwen-Image (extra run) | text-to-image | `20B` | `2025-08-02` | `512x512`, `steps=30`, `bf16` | `~3.5m` | GPU | `apache-2.0` | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) | `reports/publish/qwen_image_512.log`, `qwen-image/out/qwen_image_512_75g_retest2.png` |
| Qwen-Image | text-to-image | `20B` | `2025-08-02` | `1024x1024`, `steps=30`, `bf16` | `~17.5m` | GPU | `apache-2.0` | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) | `reports/publish/qwen_image_1024.log`, `qwen-image/out/qwen_image_1024_75g_retest2.png` |
| Qwen-Image-Edit (single-image API compatibility check) | image-to-image | `20B` | `2025-08-17` | `256x256`, `steps=20`, `strength=0.6`, `bf16` | `~22m` | GPU | `apache-2.0` | [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit) | `reports/publish/qwen_image_edit_base_256_compat_steps20_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_single_compat_256_steps20_2026-03-08.png` |
| Qwen-Image-Edit-2511 (latest, single-image) | image-to-image | `20B` | `2025-12-17` | `512x512`, `steps=20`, `bf16`, seq-offload, `max_sequence_length=128`, `true_cfg_scale=2.0`, `swap=140g` | `~9.5m` | GPU + CPU offload | `apache-2.0` | [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | `reports/publish/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.png` |
| Qwen-Image-Edit-2509 (single-image) | image-to-image | `20B` | `2025-09-22` | `512x512`, `steps=4`, `bf16`, seq-offload, `max_sequence_length=128`, `true_cfg_scale=1.0`, `swap=140g` | `~6.5m` | GPU + CPU offload | `apache-2.0` | [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | `reports/publish/qwen_image_edit_2509_single_512.log`, `qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png` |
| Qwen-Image-Edit-2511 (latest, multi-image) | image-to-image | `20B` | `2025-12-17` | `2 input images`, `512x512`, `steps=20`, `bf16`, seq-offload, `max_sequence_length=128`, `true_cfg_scale=2.0`, `seed=3456`, `swap=140g` | `~13m` | GPU + CPU offload | `apache-2.0` | [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | `reports/publish/qwen_image_edit_2511_multi_512_steps20_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps20_cfg2_seed3456_2026-03-08.png` |
| Qwen-Image-Edit-2509 (multi-image) | image-to-image | `20B` | `2025-09-22` | `2 input images`, `512x512`, `steps=20`, `bf16`, seq-offload, `max_sequence_length=128`, `true_cfg_scale=2.0`, `seed=3456`, `swap=140g` | `~13m` | GPU + CPU offload | `apache-2.0` | [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | `reports/publish/qwen_image_edit_2509_multi_512_steps20_cfg2_seed3456_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_2509_multi_512_steps20_cfg2_seed3456_2026-03-08.png` |
| FLUX.2-klein-base-4B | text-to-image | `4B` | `2026-01-14` | `512x512`, `steps=20`, `guidance=1.0`, `fp32` | `~4.5m` | GPU | `apache-2.0` | [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | `reports/publish/flux2_klein_base_4b_512_steps20_2026-03-08.log`, `stable-diffusion/out/flux2_klein_base_4b_512_steps20_2026-03-08.png` |
| FLUX.2-klein-9B | text-to-image | `9B` | `2026-01-14` | `512x512`, `steps=20`, `guidance=1.0`, `bf16`, CPU offload | `~3m` | GPU + CPU offload | `flux-non-commercial-license` | [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `reports/publish/flux2_klein_9b_512_t2i_steps20_2026-03-08.log`, `stable-diffusion/out/flux2_klein_9b_512_t2i_steps20_2026-03-08.png` |
| diffusers/FLUX.2-dev-bnb-4bit | text-to-image + image-to-image | `32B` | `2025-11-24` | t2i: `512x512`, `steps=20`, `guidance=3.0`, `bf16`, CPU offload; i2i/multi: `512x512`, `steps=4` | `~4m` (t2i) | GPU + CPU offload | `flux-dev-non-commercial-license` | [`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) | `reports/publish/flux2_dev_bnb4_512_t2i_steps20_2026-03-08.log`, `stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps20_2026-03-08.png`, `reports/publish/flux2_dev_bnb4_512_i2i.log`, `stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png`, `reports/publish/flux2_dev_bnb4_512_multi.log`, `stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png` |
| diffusers/FLUX.2-dev-bnb-4bit (recommended-step profile) | text-to-image | `32B` | `2025-11-24` | `512x512`, `steps=30`, `guidance=3.5`, `bf16`, CPU offload (`max_sequence_length=256`, `timeout=45m`) | `~5m` | GPU + CPU offload | `flux-dev-non-commercial-license` | [`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) | `reports/publish/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.log`, `stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.png` |
| FLUX.2-dev-NVFP4 | text-to-image | `32B` | `2025-12-31` | `512x512`, `steps=4` load attempt | `~0.5m` (expected fail) | n/a (format mismatch) | `flux-dev-non-commercial-license` | [`black-forest-labs/FLUX.2-dev-NVFP4`](https://huggingface.co/black-forest-labs/FLUX.2-dev-NVFP4) | `reports/publish/flux2_dev_nvfp4_expected_fail.log` |
| SD3.5 Medium | text-to-image | `2.5B` | `2024-10-29` | `512x512`, `steps=40`, `guidance=4.5` | `~8m` | GPU | `other` | [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | `reports/publish/sd35_medium_512.log`, `stable-diffusion/out/sd35_sample_best_retest.png` |
| SD3.5 Large | text-to-image | `8.1B` | `2024-10-22` | `512x512`, `steps=20`, `guidance=3.5` | `~16m` | GPU | `other` | [`stabilityai/stable-diffusion-3.5-large`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | `reports/publish/sd35_large_512.log`, `stable-diffusion/out/sd35_large_512_best_attempt.png` |
| SDXL Base | text-to-image | `3.5B` | `2023-07-25` | `512x512`, `steps=50`, `guidance=5.0` | `~5m` | GPU | `openrail++` | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `reports/publish/sdxl_base_512.log`, `stable-diffusion/out/sdxl_base_best_retest.png` |
| SDXL Base (photoreal comparison prompt) | text-to-image | `3.5B` | `2023-07-25` | `512x512`, `steps=50`, `guidance=5.0`, photoreal prompt + negative prompt | `~4.5m` | GPU | `openrail++` | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `reports/publish/sdxl_base_512_photoreal_2026-03-08.log`, `stable-diffusion/out/sdxl_base_photoreal_512_2026-03-08.png` |
| Playground v2.5 (SDXL fine-tune) | text-to-image | `3.5B` | `2024-02-16` | `1024x1024`, `steps=20`, `guidance=3.0`, `fp16`, `vae_tiling=1` | `~11.5m` | GPU | `playground-v2dot5-community` | [`playgroundai/playground-v2.5-1024px-aesthetic`](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | `reports/publish/playground_v25_1024.log`, `stable-diffusion/out/playground_v25_1024_2026-02-12.png` |
| FLUX.2-klein-4B | text-to-image | `4B` | `2026-01-14` | `512x512`, `steps=4`, `guidance=1.0` | `~3m` | GPU | `apache-2.0` | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `reports/publish/flux2_klein_4b_512.log`, `stable-diffusion/out/flux2_klein_best_retest.png` |
| SD x4 Upscaler | upscaler | `0.869B` | `2022-11-23` | `512 -> 2048`, `steps=8`, `guidance=6.0`, `noise=10`, `fp16`, `aotriton`, `sdp` | `~21m` | GPU | `openrail++` | [`stabilityai/stable-diffusion-x4-upscaler`](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) | `reports/publish/sd_x4_upscale_512_to_2048.log`, `stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png`, `reports/research/sd_x4_upscaler_param_count.json` |
| Qwen2.5-VL-7B | image-to-text | `7B` | `2025-01-26` | `image=512x512`, `max_new_tokens=256` | `~1.5m` | CPU fallback | `apache-2.0` | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `reports/publish/qwen_vl_7b_cpu.log`, `qwen-vl/out/qwen_vl_describe_75g_retest2.txt` |
| Gemma 4 26B-A4B (GGUF, Q4_K_M) | vlm + mcp tool-use (chat + coding + image description + detection JSON + Playwright/wiki + docker-shell image ops) | `26B total / 4B active` | `2026-04-02` | task suite (`ctx=32768`, `max_tokens=768`, `image=512x512`) + context sweep max (`ctx=262144`, `max_tokens=256`) + compaction demo (coding + non-coding) + MCP tool-use demo (`4` Wikipedia painting pages -> docker resize + contact sheet) | `~1m` (suite) + `~5m` (ctx sweep) + `~1m` (compaction) + `~1m` (MCP demo) | GPU (Vulkan) | `apache-2.0` | [`google/gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26b-a4b-it) + [`ggml-org/gemma-4-26b-a4b-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-26b-a4b-it-GGUF) | `reports/publish/gemma4_vlm_task_suite_q4_2026-04-03_r3.log`, `llama-cpp-vulkan/out/gemma4-task-suite/gemma-4-26B-A4B-it-Q4_K_M_2026-04-03q4r3_vision_describe.json`, `llama-cpp-vulkan/out/gemma4-task-suite/gemma-4-26B-A4B-it-Q4_K_M_2026-04-03q4r3_vision_detect.json`, `reports/publish/gemma4_ctx_text_matrix_2026-04-03.tsv`, `reports/publish/gemma4_context_compaction_q4_2026-04-03_r2.log`, `reports/publish/gemma4_mcp_wiki_paintings_2026-04-03_r10.log`, `mcp/out/wiki_paintings_2026-04-03-r10/summary.json` |
| Qwen3.5-9B Q8_0 (GGUF) | vlm (vision + summarization + coding) | `9B` | `2026-02-28` | task suite (`image=512x512`, summarization prompt, coding prompt), `ctx=32768`; simple-text ctx probe max: `ctx=131072`, `max_tokens=256` | `~1.5m` (suite) + `~1m` (ctx probe) | GPU (Vulkan) | `apache-2.0` | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `reports/publish/qwen35_9b_q8_task_suite_2026-03-08b.log`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_9b_q8_2026_03_08b_vision.json`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_9b_q8_2026_03_08b_summarization.json`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_9b_q8_2026_03_08b_coding.json`, `reports/publish/qwen35_ctx_text/qwen35_9b_q8_ctx131072.log`, `llama-cpp-vulkan/out/qwen35-ctx-text/qwen35_9b_q8_ctx131072_2026-03-08.json` |
| Qwen3.5-27B Q8_0 (GGUF) | vlm (vision + summarization + coding) | `27B` | `2026-02-24` | task suite (`image=512x512`, summarization prompt, coding prompt), `ctx=16384`; simple-text ctx probe max: `ctx=131072`, `max_tokens=256` | `~4m` (suite) + `~1.5m` (ctx probe) | GPU (Vulkan) | `apache-2.0` | [`unsloth/Qwen3.5-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | `reports/publish/qwen35_27b_q8_task_suite_2026-03-08b.log`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_27b_q8_2026_03_08b_vision.json`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_27b_q8_2026_03_08b_summarization.json`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_27b_q8_2026_03_08b_coding.json`, `reports/publish/qwen35_ctx_text/qwen35_27b_q8_ctx131072.log`, `llama-cpp-vulkan/out/qwen35-ctx-text/qwen35_27b_q8_ctx131072_2026-03-08.json` |
| Qwen3.5-122B-A10B Q4_K_M (GGUF) | vlm (vision + summarization + coding) | `122B total / 10B active` | `2026-02-24` | task suite (`image=512x512`, summarization prompt, coding prompt), `ctx=8192`; simple-text ctx probe max: `ctx=131072`, `max_tokens=128` | `~6.5m` (suite) + `~2m` (ctx probe) | CPU fallback | `apache-2.0` | [`unsloth/Qwen3.5-122B-A10B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) | `reports/publish/qwen35_122b_a10b_q4km_task_suite_2026-03-08b.log`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_122b_a10b_q4km_2026_03_08b_vision.json`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_122b_a10b_q4km_2026_03_08b_summarization.json`, `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_122b_a10b_q4km_2026_03_08b_coding.json`, `reports/publish/qwen35_ctx_text/qwen35_122b_a10b_q4km_ctx131072.log`, `llama-cpp-vulkan/out/qwen35-ctx-text/qwen35_122b_a10b_q4km_ctx131072_2026-03-08.json` |
| VGGT-1B | 3d-reconstruction | `1B` | `2025-03-11` | `12 photos -> 200k-point .ply`, `load=768`, `model=448` | `~3m` | GPU | `cc-by-nc-4.0` | [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) | `reports/publish/vggt_reconstruct.log`, `reconstruction-3d/out/south_building/south_building_points.ply` |
| YOLO26n (detect) | vision-detection | `~0.004B` | `n/a` | `bus.jpg`, `conf=0.25`, `imgsz=640` | `~0.5m` | CPU | `agpl-3.0` | [`Ultralytics YOLO26`](https://docs.ultralytics.com/models/yolo26/) | `reports/publish/yolo26n_detect_bus.log`, `vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg`, `vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json` |
| YOLO26n-pose (keypoints) | vision-pose | `~0.004B` | `n/a` | `bus.jpg`, `conf=0.25`, `imgsz=640` | `~0.5m` | CPU | `agpl-3.0` | [`Ultralytics Pose Task`](https://docs.ultralytics.com/tasks/pose/) | `reports/publish/yolo26n_pose_bus.log`, `vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg`, `vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json` |
| UltraFace + ArcFace (ONNX) | face-recognition | `~0.066B + ~0.0003B` | `2025-09-29` | `2x2` synthetic collage (`4` identities), `conf=0.7`, `crop_margin=0.25` | `~0.5m` | CPU | `apache-2.0` | [`onnxmodelzoo/arcfaceresnet100-8`](https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8) + [`onnxmodelzoo/version-RFB-320`](https://huggingface.co/onnxmodelzoo/version-RFB-320) | `reports/publish/vision_face_match_demo.log`, `vision-face/out/face_match_annotated.png`, `vision-face/out/face_match_summary.json` |
| Qwen3-Next-80B-A3B | llm-chat | `80B total / 3B active` | `2025-12-03` | `ctx=196608`, `max_tokens=1024` | `~1.5m` | GPU | `apache-2.0` | [`Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF) | `reports/publish/llama_qwen3_next_ctx196608.log`, `llama-cpp-vulkan/out/qwen3_next_80b_q5_ctx_196608_75g_retest2.json` |
| Qwen3-Coder-Next-Q5 | llm-coding | `80B total / 3B active` | `2026-02-02` | `ctx=196608`, `max_tokens=1536` | `~2.5m` | GPU | `apache-2.0` | [`Qwen/Qwen3-Coder-Next-GGUF`](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | `reports/publish/llama_qwen3_coder_next_ctx196608.log`, `llama-cpp-vulkan/out/qwen3_coder_next_q5_ctx_196608_75g_retest2.json` |
| Qwen2.5-Coder-32B | llm-coding | `32B` | `2024-11-09` | `ctx=131072`, `max_tokens=1536` | `~1.5m` | GPU | `apache-2.0` | [`Qwen/Qwen2.5-Coder-32B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) | `reports/publish/llama_qwen25_coder_32b_ctx131072.log`, `llama-cpp-vulkan/out/qwen25_coder_32b_q4_ctx_131072_75g_retest2.json` |
| GPT-OSS-120B MXFP4 | llm-chat | `120B` | `2025-08-02` | `ctx=131072`, `max_tokens=1024` | `~2m` | GPU | `apache-2.0` | [`ggml-org/gpt-oss-120b-GGUF`](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) | `reports/publish/llama_gpt_oss_120b_ctx131072.log`, `llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_ctx_131072_75g_retest2.json` |
| Kokoro-82M | audio-tts | `0.082B` | `2024-12-26` | transcript -> `~157.9s` waveform, `voice=af_heart` | `~1.5m` | CPU | `apache-2.0` | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) | `reports/publish/audio_kokoro_tts.log`, `audio/out/podcast_kokoro_best_retest.wav` |
| faster-whisper (small) | audio-stt | `0.244B` | `2022-09-26` | `~157.9s` audio -> `.srt` (`39` segments) | `~2m` | CPU | `apache-2.0` | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | `reports/publish/audio_faster_whisper_subtitles.log`, `audio/out/podcast_kokoro_best_retest.srt` |
| Voxtral-Mini-3B-2507 | audio-stt | `3B` | `2025-07-01` | `30s` clip transcription, `max_new_tokens=512` | `~0.5m` | GPU | `apache-2.0` | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | `reports/publish/audio_voxtral_mini_3b_transcribe.log`, `audio/out/voxtral_mini_3b_2507_transcript.txt` |
| Wan2.1-T2V-1.3B | video-gen | `1.3B` | `2025-03-01` | `672x384`, `17` frames, `steps=8`, `fps=8` | `~7.5m` | GPU + CPU offload | `apache-2.0` | [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | `reports/publish/wan21_t2v.log`, `video/out/wan21_t2v_sample.mp4` |

Non-model workflow retests (MCP + agentic demo) are tracked in:
- `reports/publish/summary_final.tsv` (see `category=mcp` and `category=agentic`)

License note:
- `ggml-org/gpt-oss-120b-GGUF` does not currently expose a `cardData.license` field in the HF API response, so `apache-2.0` is taken from the upstream base model card (`openai/gpt-oss-120b`).

Set this once in your shell before running commands:

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
cd "$REPO_ROOT"
```

## Text to Image

The sections below are intentionally self-contained: each model write-up includes what it does, how to reproduce the exact run (scripts + Docker image tag), and links to the saved evidence artifacts.
All commands assume you already ran the env snippet above (`export REPO_ROOT=...` + `source scripts/env.sh`).

### Qwen-Image-2512 (latest)

Text-to-image diffusion model by Qwen (see the summary table above for size/license/date).

Model card: [`Qwen/Qwen-Image-2512`](https://huggingface.co/Qwen/Qwen-Image-2512)
Docker image: `qwen-image-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512x512`, `steps=20`, `bf16`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/qwen-image-2512 WIDTH=512 HEIGHT=512 STEPS=20 DTYPE=bfloat16 \
      OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_2512_512_2026-02-11.png \
  bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh
```

Evidence:
- `reports/publish/qwen_image_2512_512.log`
- `qwen-image/out/qwen_image_2512_512_2026-02-11.png`

![Qwen-Image-2512 sample](qwen-image/out/qwen_image_2512_512_2026-02-11.png)
Caption: `Qwen-Image-2512` | `512x512` | `steps=20` | `bf16`

### Qwen-Image (512 and 1024)

Older Qwen text-to-image release (kept because it still works well and supports `1024x1024` on this host).

Model card: [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image)
Docker image: `qwen-image-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512x512`, `steps=30`, `bf16`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/qwen-image WIDTH=512 HEIGHT=512 STEPS=30 DTYPE=bfloat16 \
      OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png \
  bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh
```

Reproduce (validated `1024x1024`, `steps=30`, `bf16`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/qwen-image WIDTH=1024 HEIGHT=1024 STEPS=30 DTYPE=bfloat16 CURL_MAX_TIME=2400 \
      OUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_1024_75g_retest2.png \
  bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh
```

Evidence:
- `reports/publish/qwen_image_512.log`, `qwen-image/out/qwen_image_512_75g_retest2.png`
- `reports/publish/qwen_image_1024.log`, `qwen-image/out/qwen_image_1024_75g_retest2.png`

![Qwen-Image 512 sample](qwen-image/out/qwen_image_512_75g_retest2.png)
Caption: `Qwen-Image` | `512x512` | `steps=30` | `bf16`

![Qwen-Image 1024 sample](qwen-image/out/qwen_image_1024_75g_retest2.png)
Caption: `Qwen-Image` | `1024x1024` | `steps=30` | `bf16`

### FLUX.2-klein-base-4B

Modern diffusion baseline (small enough to run comfortably; good quality per step count).

Model card: [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512x512`, `steps=20`, `guidance=1.0`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-klein-base-4b DTYPE=float32 HEIGHT=512 WIDTH=512 STEPS=20 GUIDANCE=1.0 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_base_4b_512_steps20_2026-03-08.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_sample.sh
```

Evidence:
- `reports/publish/flux2_klein_base_4b_512_steps20_2026-03-08.log`
- `stable-diffusion/out/flux2_klein_base_4b_512_steps20_2026-03-08.png`

![FLUX.2-klein-base-4B sample](stable-diffusion/out/flux2_klein_base_4b_512_steps20_2026-03-08.png)
Caption: `FLUX.2-klein-base-4B` | `512x512` | `steps=20` | `guidance=1.0`

### FLUX.2-klein-4B

Klein family variant used as a quick “modern diffusion” reference point.

Model card: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512x512`, `steps=4`, `guidance=1.0`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-klein-4b DTYPE=float32 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=1.0 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_best_retest.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_sample.sh
```

Evidence:
- `reports/publish/flux2_klein_4b_512.log`
- `stable-diffusion/out/flux2_klein_best_retest.png`

![FLUX.2-klein-4B sample](stable-diffusion/out/flux2_klein_best_retest.png)
Caption: `FLUX.2-klein-4B` | `512x512` | `steps=4` | `guidance=1.0`

### FLUX.2-klein-9B

Larger Klein model (gated on HF). On this host it requires CPU offload, but it can still complete `512x512` under the strict `75g` policy.

Model card: [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Download (gated; requires HF access):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/stable-diffusion/scripts/download_flux2_klein_9b.sh
```

Reproduce (validated `512x512`, `steps=20`, `guidance=1.0`, CPU offload):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-klein-9b HEIGHT=512 WIDTH=512 STEPS=20 GUIDANCE=1.0 MODEL_CPU_OFFLOAD=1 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_klein_9b_512_t2i_steps20_2026-03-08.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_klein_probe.sh
```

Evidence:
- `reports/publish/flux2_klein_9b_512_t2i_steps20_2026-03-08.log`
- `stable-diffusion/out/flux2_klein_9b_512_t2i_steps20_2026-03-08.png`

![FLUX.2-klein-9B sample](stable-diffusion/out/flux2_klein_9b_512_t2i_steps20_2026-03-08.png)
Caption: `FLUX.2-klein-9B` | `512x512` | `steps=20` | `guidance=1.0` | `bf16 + CPU offload`

### FLUX.2-dev-bnb-4bit (diffusers)

Practical “dev-family” option on this host (gated on HF). Validated here for text-to-image, image-to-image, and multi-image composition.

Model card: [`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Download (gated; requires HF access):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/stable-diffusion/scripts/download_flux2_dev_bnb4.sh
```

Reproduce (validated text-to-image `512x512`, `steps=20`, `guidance=3.0`, CPU offload):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=20 GUIDANCE=3.0 DTYPE=bfloat16 \
      MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps20_2026-03-08.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh
```

Reproduce (validated image-to-image `512x512`, `steps=4`, `guidance=3.0`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=3.0 DTYPE=bfloat16 \
      MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 \
      INIT_IMAGE=qwen-image/out/qwen_image_512_75g_retest2.png \
      PROMPT='convert this into a cinematic oil painting while preserving composition' \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh
```

Reproduce (validated multi-image `512x512`, `steps=4`, `guidance=3.0`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=4 GUIDANCE=3.0 DTYPE=bfloat16 \
      MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=128 \
      INIT_IMAGE=qwen-image/out/qwen_image_512_75g_retest2.png,qwen-image/out/qwen_image_2512_512_2026-02-11.png \
      PROMPT='blend both reference images into one coherent cinematic scene with realistic lighting' \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh
```

Evidence:
- `reports/publish/flux2_dev_bnb4_512_t2i_steps20_2026-03-08.log`, `stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps20_2026-03-08.png`
- `reports/publish/flux2_dev_bnb4_512_i2i.log`, `stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png`
- `reports/publish/flux2_dev_bnb4_512_multi.log`, `stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png`

![FLUX.2-dev-bnb4 sample](stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps20_2026-03-08.png)
Caption: `FLUX.2-dev-bnb-4bit` (t2i) | `512x512` | `steps=20` | `guidance=3.0` | `bf16 + CPU offload`

![FLUX.2-dev-bnb4 i2i sample](stable-diffusion/out/flux2_dev_bnb4_512_i2i_reconfirm_2026-02-12.png)
Caption: `FLUX.2-dev-bnb-4bit` (i2i) | `512x512` | `steps=4` | `guidance=3.0` | init: `qwen-image/out/qwen_image_512_75g_retest2.png`

![FLUX.2-dev-bnb4 multi sample](stable-diffusion/out/flux2_dev_bnb4_512_multi_reconfirm_2026-02-12.png)
Caption: `FLUX.2-dev-bnb-4bit` (multi) | `512x512` | `steps=4` | `guidance=3.0` | init: two Qwen-Image outputs

Recommended-step profile run (newer quality-focused run):

Official references:
- BFL model card suggests `num_inference_steps=50` for FLUX.2-dev quality: [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev).
- Official HF Space uses `28` as a practical speed/quality profile: [`black-forest-labs/FLUX.2-dev` Space app](https://huggingface.co/spaces/black-forest-labs/FLUX.2-dev/blob/main/app.py).

This repo now includes an in-between profile (`steps=30`) with a `45m` hard timeout for stability.

```bash
timeout 2700 $REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/flux2-dev-bnb4 HEIGHT=512 WIDTH=512 STEPS=30 GUIDANCE=3.5 DTYPE=bfloat16 \
      MODEL_CPU_OFFLOAD=1 USE_REMOTE_TEXT_ENCODER=0 MAX_SEQUENCE_LENGTH=256 SEED=42 \
      PROMPT='A cinematic, photoreal portrait of a friendly humanoid robot barista pouring latte art in a warm coffee shop, 85mm lens, shallow depth of field, natural skin-like materials, high detail, realistic lighting' \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_flux2_dev_bnb4_probe.sh \
  > $REPO_ROOT/reports/publish/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.log 2>&1
```

Evidence:
- `reports/publish/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.log`
- `stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.png`

![FLUX.2-dev-bnb4 30-step sample](stable-diffusion/out/flux2_dev_bnb4_512_t2i_steps30_2026-03-07.png)
Caption: `FLUX.2-dev-bnb-4bit` (t2i, recommended-step profile) | `512x512` | `steps=30` | `guidance=3.5` | `bf16 + CPU offload` | `timeout=45m`

### FLUX.2-dev-NVFP4 (fails in this stack)

`FLUX.2-dev-NVFP4` downloads, but this repo’s diffusers container expects a full diffusers directory (it does not load checkpoint-only repos directly).

Model card: [`black-forest-labs/FLUX.2-dev-NVFP4`](https://huggingface.co/black-forest-labs/FLUX.2-dev-NVFP4)

Evidence:
- `reports/publish/flux2_dev_nvfp4_expected_fail.log`

### Stable Diffusion 3.5 Medium

Model card: [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512x512`, `steps=40`, `guidance=4.5`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_sample_best_retest.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_sample.sh
```

Evidence:
- `reports/publish/sd35_medium_512.log`
- `stable-diffusion/out/sd35_sample_best_retest.png`

![SD3.5 Medium sample](stable-diffusion/out/sd35_sample_best_retest.png)
Caption: `SD3.5 Medium` | `512x512` | `steps=40` | `guidance=4.5`

### Stable Diffusion 3.5 Large

Model card: [`stabilityai/stable-diffusion-3.5-large`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512x512`, `steps=20`, `guidance=3.5`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env WIDTH=512 HEIGHT=512 STEPS=20 GUIDANCE=3.5 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/sd35_large_512_best_attempt.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_sd35_large_sample.sh
```

Evidence:
- `reports/publish/sd35_large_512.log`
- `stable-diffusion/out/sd35_large_512_best_attempt.png`

![SD3.5 Large sample](stable-diffusion/out/sd35_large_512_best_attempt.png)
Caption: `SD3.5 Large` | `512x512` | `steps=20` | `guidance=3.5`

### SDXL Base

Model card: [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).
Prompting references used for this comparison run:
- SDXL is 1024-native (official docs), but this repo keeps `512x512` here for fair runtime/workload comparison with other 512 tests: [`Diffusers SDXL guide`](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl).
- Diffusers supports prompt + negative prompt conditioning in the same generation call: [`Diffusers text-to-image API`](https://huggingface.co/docs/diffusers/main/en/using-diffusers/text2img).

Reproduce (validated `512x512`, `steps=50`, `guidance=5.0`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env OUT_PATH=$REPO_ROOT/stable-diffusion/out/sdxl_base_best_retest.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_sdxl_base_sample.sh
```

Reproduce (photoreal comparison run, same workload):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env PROMPT='Photorealistic golden-hour panorama of a dramatic alpine lake valley, rugged mountain ridges, crystal water reflections, pine trees in foreground, natural atmospheric haze, realistic color grading, full-frame DSLR photo, 35mm lens, high dynamic range, ultra-detailed textures.' \
      NEGATIVE_PROMPT='cartoon, cgi, illustration, anime, painting, plastic skin, oversaturated, blurry, low detail, deformed geometry, watermark, text, logo' \
      STEPS=50 GUIDANCE=5.0 HEIGHT=512 WIDTH=512 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/sdxl_base_photoreal_512_2026-03-08.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_sdxl_base_sample.sh \
  > $REPO_ROOT/reports/publish/sdxl_base_512_photoreal_2026-03-08.log 2>&1
```

Evidence:
- `reports/publish/sdxl_base_512.log`
- `stable-diffusion/out/sdxl_base_best_retest.png`
- `reports/publish/sdxl_base_512_photoreal_2026-03-08.log`
- `stable-diffusion/out/sdxl_base_photoreal_512_2026-03-08.png`

![SDXL Base sample](stable-diffusion/out/sdxl_base_best_retest.png)
Caption: `SDXL Base` | `512x512` | `steps=50` | `guidance=5.0`

![SDXL Base photoreal sample](stable-diffusion/out/sdxl_base_photoreal_512_2026-03-08.png)
Caption: `SDXL Base` (photoreal comparison prompt) | `512x512` | `steps=50` | `guidance=5.0` | prompt tuned for photographic scene realism

### Playground v2.5 (SDXL fine-tune)

High-quality SDXL fine-tune with good instruction-following at `1024x1024`.

Model card: [`playgroundai/playground-v2.5-1024px-aesthetic`](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `1024x1024`, `steps=20`, `guidance=3.0`, `fp16`, `vae_tiling=1`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env OUT_PATH=$REPO_ROOT/stable-diffusion/out/playground_v25_1024_2026-02-12.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_playground_v25_sample.sh
```

Evidence:
- `reports/publish/playground_v25_1024.log`
- `stable-diffusion/out/playground_v25_1024_2026-02-12.png`

![Playground v2.5 sample](stable-diffusion/out/playground_v25_1024_2026-02-12.png)
Caption: `Playground v2.5` | `1024x1024` | `steps=20` | `guidance=3.0` | `fp16` | `vae_tiling=1`

## Image to Image

### Qwen-Image-Edit

Qwen image editing (single-image input). This is the “base” edit model (not the Plus 2509/2511 family).

Model card: [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit)
Docker image: `qwen-image-edit-rocm:latest` (build: see [Containers](#containers)).

Reproduce (single-image API compatibility check after multi-image API update; `256x256`, `steps=20`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_ID=$MODEL_ROOT/qwen-image-edit INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png \
      HEIGHT=256 WIDTH=256 STEPS=20 STRENGTH=0.6 DTYPE=bfloat16 \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_single_compat_256_steps20_2026-03-08.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit.sh
```

Evidence:
- `reports/publish/qwen_image_edit_base_256_compat_steps20_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_single_compat_256_steps20_2026-03-08.png`

![Qwen-Image-Edit compat sample](qwen-image-edit/out/qwen_image_edit_single_compat_256_steps20_2026-03-08.png)
Caption: `Qwen-Image-Edit` (compat check) | `256x256` | `steps=20` | `strength=0.6` | `bf16`

### Qwen-Image-Edit-2509

“Plus” edit model (supports multi-image composition). On this host, stable runs required sequential CPU offload and a larger swap budget.

Model card: [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
Docker image: `qwen-image-edit-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated single-image `512x512`, `steps=4`, sequential CPU offload, `bf16`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=single MODEL_ID=$MODEL_ROOT/qwen-image-edit-2509 INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png \
      HEIGHT=512 WIDTH=512 STEPS=4 MEMORY_SWAP=140g \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh
```

Reproduce (validated multi-image `512x512`, `steps=20`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=multi MODEL_ID=$MODEL_ROOT/qwen-image-edit-2509 HEIGHT=512 WIDTH=512 STEPS=20 TRUE_CFG_SCALE=2.0 SEED=3456 MEMORY_SWAP=140g \
      INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_a_512_seed1234.png \
      INPUT_IMAGE_B=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png \
      PROMPT='Take the person from image A and insert them into image B. Keep the original person from image B. Place the inserted person on the left side of image B, standing naturally near the other person. Preserve both faces, match lighting and perspective, keep the coffee shop background unchanged, sharp focus.' \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2509_multi_512_steps20_cfg2_seed3456_2026-03-08.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh
```

Evidence:
- `reports/publish/qwen_image_edit_2509_single_512.log`, `qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png`
- `reports/publish/qwen_image_edit_2509_multi_512_steps20_cfg2_seed3456_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_2509_multi_512_steps20_cfg2_seed3456_2026-03-08.png`

![Qwen-Image-Edit-2509 single sample](qwen-image-edit/out/qwen_image_edit_2509_single_512_seqoffload_bf16_75g_test.png)
Caption: `Qwen-Image-Edit-2509` (single) | `512x512` | `steps=4` | `bf16 + sequential CPU offload`

![Qwen-Image-Edit-2509 multi sample](qwen-image-edit/out/qwen_image_edit_2509_multi_512_steps20_cfg2_seed3456_2026-03-08.png)
Caption: `Qwen-Image-Edit-2509` (multi) | `512x512` | `steps=20` | `true_cfg_scale=2.0` | `seed=3456` | `bf16 + sequential CPU offload`

### Qwen-Image-Edit-2511

Latest “Plus” edit model (supports multi-image composition). Like 2509, stable runs on this host required sequential CPU offload and a larger swap budget.

Model card: [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
Docker image: `qwen-image-edit-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated single-image `512x512`, `steps=20`, sequential CPU offload, `bf16`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=single MODEL_ID=$MODEL_ROOT/qwen-image-edit-2511 \
      INPUT_IMAGE=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png \
      PROMPT='Edit this single image only: keep the same robot barista and same coffee shop scene, remove any extra background objects, keep one subject, improve sharpness and latte details.' \
      HEIGHT=512 WIDTH=512 STEPS=20 TRUE_CFG_SCALE=2.0 MEMORY_SWAP=140g \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh
```

Reproduce (validated multi-image “move person A into photo B”, `512x512`, `steps=20`, `cfg=2.0`, `seed=3456`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODE=multi MODEL_ID=$MODEL_ROOT/qwen-image-edit-2511 HEIGHT=512 WIDTH=512 STEPS=20 TRUE_CFG_SCALE=2.0 SEED=3456 MEMORY_SWAP=140g \
      INPUT_IMAGE_A=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_a_512_seed1234.png \
      INPUT_IMAGE_B=$REPO_ROOT/qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png \
      PROMPT='Take the person from image A and insert them into image B. Keep the original person from image B. Place the inserted person on the left side of image B, standing naturally near the other person. Preserve both faces, match lighting and perspective, keep the coffee shop background unchanged, sharp focus.' \
      OUT_PATH=$REPO_ROOT/qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps20_cfg2_seed3456_2026-03-08.png \
  bash $REPO_ROOT/qwen-image-edit/scripts/test_qwen_image_edit_plus_stable.sh
```

Evidence:
- `reports/publish/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.png`
- `reports/publish/qwen_image_edit_2511_multi_512_steps20_2026-03-08.log`, `qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps20_cfg2_seed3456_2026-03-08.png`

![Qwen-Image-Edit-2511 single sample](qwen-image-edit/out/qwen_image_edit_2511_single_512_steps20_clean_2026-03-08.png)
Caption: `Qwen-Image-Edit-2511` (single) | `512x512` | `steps=20` | `true_cfg_scale=2.0` | `bf16 + sequential CPU offload` | single-image prompt (no multi-image composition)

![Qwen-Image-Edit-2511 multi input A](qwen-image-edit/input/qwen_image_2512_person_a_512_seed1234.png)
Caption: input A | `Qwen-Image-2512` | `512x512` | `steps=20` | `seed=1234`

![Qwen-Image-Edit-2511 multi input B](qwen-image-edit/input/qwen_image_2512_person_b_512_seed2345.png)
Caption: input B | `Qwen-Image-2512` | `512x512` | `steps=20` | `seed=2345`

![Qwen-Image-Edit-2511 multi sample](qwen-image-edit/out/qwen_image_edit_2511_multi_move_person_512_steps20_cfg2_seed3456_2026-03-08.png)
Caption: `Qwen-Image-Edit-2511` (multi) | `512x512` | `steps=20` | `true_cfg_scale=2.0` | `seed=3456` | `bf16 + sequential CPU offload`

Note: `FLUX.2-dev-bnb-4bit` also supports i2i and multi-image composition. See [FLUX.2-dev-bnb-4bit (diffusers)](#flux2-dev-bnb-4bit-diffusers).

## Upscaling

### SD x4 Upscaler

Diffusion upscaler used here to reduce the “plastic” feel on some generated outputs and to validate large output (`2048x2048`) under the strict memory policy.

Model card: [`stabilityai/stable-diffusion-x4-upscaler`](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)
Docker image: `stable-diffusion-rocm:latest` (build: see [Containers](#containers)).

Reproduce (validated `512 -> 2048`, `steps=8`, `guidance=6.0`, `noise_level=10`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env INPUT_PATH=$REPO_ROOT/qwen-image/out/qwen_image_1024_75g_retest2.png INPUT_SIDE=512 \
      STEPS=8 NOISE_LEVEL=10 GUIDANCE=6.0 DTYPE=float16 DEVICE=cuda DISABLE_SDP=0 AOTRITON_EXPERIMENTAL=1 \
      OUT_PATH=$REPO_ROOT/stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png \
  bash $REPO_ROOT/stable-diffusion/scripts/test_sd_x4_upscale_qwen_image.sh
```

Evidence:
- `reports/publish/sd_x4_upscale_512_to_2048.log`
- `stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png`

![Qwen-Image x4 upscaled sample](stable-diffusion/out/qwen_image_upscaled_2048_best_retest.png)
Caption: `SD x4 Upscaler` | input `512 -> 2048` | `steps=8` | `guidance=6.0` | `noise_level=10`

## Image to Text

### Qwen2.5-VL-7B-Instruct

Vision-language model used here as a CPU fallback captioner for debugging image edits and validating that image-to-text works even without a stable ROCm VLM stack.

Model card: [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
Docker image: `qwen-vl-rocm:latest` (build: see [Containers](#containers)).

Reproduce (CPU fallback, `max_new_tokens=256`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env FORCE_CPU=1 MAX_NEW_TOKENS=256 \
      INPUT_IMAGE=$REPO_ROOT/qwen-image/out/qwen_image_512_75g_retest2.png \
      OUT_PATH=$REPO_ROOT/qwen-vl/out/qwen_vl_describe_75g_retest2.txt \
  bash $REPO_ROOT/qwen-vl/scripts/test_qwen_vl_7b.sh
```

Evidence:
- `reports/publish/qwen_vl_7b_cpu.log`
- `qwen-vl/out/qwen_vl_describe_75g_retest2.txt`

### Qwen3.5 VLM suite (cross-links)

The Qwen3.5 models were validated with a 3-task suite that includes image-to-text (`vision.json`) plus summarization and coding outputs.

- [Qwen3.5-9B Q8_0 (GGUF, VLM)](#qwen35-9b-q8_0-gguf-vlm)
- [Qwen3.5-27B Q8_0 (GGUF, VLM)](#qwen35-27b-q8_0-gguf-vlm)
- [Qwen3.5-122B-A10B Q4_K_M (GGUF, VLM)](#qwen35-122b-a10b-q4_k_m-gguf-vlm)

### Gemma 4 26B-A4B (cross-links)

Gemma 4 was validated with the same practical VLM pattern used in this repo:
- chat/QA
- coding generation
- image description
- object-detection-style JSON output

See full details in [Gemma 4 26B-A4B (GGUF, VLM)](#gemma-4-26b-a4b-gguf-vlm).

## Detection and Pose

### YOLO26n (detect)

Local object/person detection with `xyxy` bounding boxes and confidence scores (JSON output is designed for downstream automation).

Docs: [`Ultralytics YOLO26`](https://docs.ultralytics.com/models/yolo26/)
Docker image: `vision-yolo-rocm:latest` (build: see `vision-detection/README.md`).

Reproduce (CPU, `imgsz=640`, `conf=0.25`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_NAME=yolo26n.pt DEVICE=cpu CONF=0.25 \
      INPUT_IMAGE=$REPO_ROOT/vision-detection/input/bus.jpg \
      OUT_JSON=$REPO_ROOT/vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json \
      OUT_IMAGE=$REPO_ROOT/vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg \
  bash $REPO_ROOT/vision-detection/scripts/test_yolo_detect.sh
```

Evidence:
- `reports/publish/yolo26n_detect_bus.log`
- `vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.json`
- `vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg`

![YOLO26 detect sample](vision-detection/out/yolo26n_detect_bus_postpatch2_2026-02-11.jpg)
Caption: `YOLO26n detect` | `bus.jpg` | `imgsz=640` | `conf=0.25` | output JSON includes `xyxy` boxes

### YOLO26n-pose

Human pose extraction with `17` keypoints per person (coordinates + confidences).

Docs: [`Ultralytics Pose Task`](https://docs.ultralytics.com/tasks/pose/)
Docker image: `vision-yolo-rocm:latest` (build: see `vision-detection/README.md`).

Reproduce (CPU, `imgsz=640`, `conf=0.25`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_NAME=yolo26n-pose.pt DEVICE=cpu CONF=0.25 \
      INPUT_IMAGE=$REPO_ROOT/vision-detection/input/bus.jpg \
      OUT_JSON=$REPO_ROOT/vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json \
      OUT_IMAGE=$REPO_ROOT/vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg \
  bash $REPO_ROOT/vision-detection/scripts/test_yolo_pose.sh
```

Evidence:
- `reports/publish/yolo26n_pose_bus.log`
- `vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.json`
- `vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg`

![YOLO26 pose sample](vision-detection/out/yolo26n_pose_bus_postpatch2_2026-02-11.jpg)
Caption: `YOLO26n-pose` | `bus.jpg` | `imgsz=640` | `conf=0.25` | output JSON includes `17` keypoints/person

## Face Recognition

### UltraFace and ArcFace (ONNX)

CPU-only face detection + embedding + nearest-neighbor matching against a small synthetic reference set.
This is safe to publish because it does **not** attempt to identify real people.

Model cards: [`onnxmodelzoo/version-RFB-320`](https://huggingface.co/onnxmodelzoo/version-RFB-320) + [`onnxmodelzoo/arcfaceresnet100-8`](https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8)
Docker image: `vision-face-onnx:latest` (build: see `vision-face/README.md`).

Reproduce (CPU):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/vision-face/scripts/test_face_match.sh
```

Evidence:
- `reports/publish/vision_face_match_demo.log`
- `vision-face/out/face_match_annotated.png`
- `vision-face/out/face_match_results.json`
- `vision-face/out/face_match_summary.json`

![Face match annotated](vision-face/out/face_match_annotated.png)
Caption: synthetic `2x2` collage | `hit_rate=0.75` (`3/4`) | CPU-only ONNX

## Coding and General Chat

### Gemma 4 26B-A4B (GGUF, VLM)

Latest Gemma multimodal MoE release (`2026-04-02`), validated here with:
1. simple chat/QA
2. coding generation
3. image description
4. object-detection-style JSON output

Model cards:
- [`google/gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26b-a4b-it)
- [`ggml-org/gemma-4-26b-a4b-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-26b-a4b-it-GGUF)

Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Download + checksum verification (GGUF + mmproj):

```bash
DOWNLOAD_Q8=1 DOWNLOAD_Q4=1 USE_ARIA2=1 \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/download_gemma4_26b_a4b_gguf.sh
```

Task suite reproduce (`Q4_K_M`, `ctx=32768`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf \
      MMPROJ_PATH=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/mmproj-gemma-4-26B-A4B-it-f16.gguf \
      PORT=8151 CTX_SIZE=32768 MAX_TOKENS=768 THREADS=8 GPU_LAYERS=999 \
      RUN_TAG=2026-04-03q4r3 \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_gemma4_vlm_task_suite.sh
```

Context sweep reproduce (`32k`, `64k`, `128k`, `168k`, `256k`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PRIMARY=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf \
      MODEL_FALLBACK=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf \
      PRIMARY_LABEL=q4 FALLBACK_LABEL=q4 \
      DATE_TAG=2026-04-03 \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/check_gemma4_ctx_matrix.sh
```

Context-compaction experiment reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf \
      MMPROJ_PATH=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/mmproj-gemma-4-26B-A4B-it-f16.gguf \
      PORT=8152 RUN_TAG=2026-04-03-r2 \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_gemma4_context_compaction_demo.sh
```

Key outcomes on this host:
- Task suite passed (chat + coding + image description + detection JSON).
- Context sweep passed at every requested setting up to `ctx=262144` under `MEM_LIMIT=75g`.
- Compaction demo (v2 profile) reduced prompt size while preserving keyword coverage:
  - coding scenario: `-28.97%` chars (`-28.32%` token estimate), keyword score `1.0 -> 1.0`
  - non-coding scenario: `-17.26%` chars (`-14.79%` token estimate), keyword score `1.0 -> 1.0`
- `Q8_0` reached expected file size but still failed SHA256 verification on this host after resume attempts; published runtime results remain based on verified `Q4_K_M`.
- Separate MCP/tool-use workflow with the same model also passed:
  - Playwright MCP browsing + snapshots of selected Wikipedia painting pages.
  - Gemma-generated shell plan executed through MCP shell server, but commands ran only inside a dedicated docker image (`mcp-image-tools:1.0`) mapped to an `mcp/out/` subdirectory.

Evidence:
- `reports/publish/gemma4_q4_mmproj_verify_2026-04-03.json`
- `reports/publish/gemma4_q8_q4_mmproj_verify_2026-04-03.json`
- `reports/publish/gemma4_vlm_task_suite_q4_2026-04-03_r3.log`
- `llama-cpp-vulkan/out/gemma4-task-suite/gemma-4-26B-A4B-it-Q4_K_M_2026-04-03q4r3_chat.json`
- `llama-cpp-vulkan/out/gemma4-task-suite/gemma-4-26B-A4B-it-Q4_K_M_2026-04-03q4r3_coding.json`
- `llama-cpp-vulkan/out/gemma4-task-suite/gemma-4-26B-A4B-it-Q4_K_M_2026-04-03q4r3_vision_describe.json`
- `llama-cpp-vulkan/out/gemma4-task-suite/gemma-4-26B-A4B-it-Q4_K_M_2026-04-03q4r3_vision_detect.json`
- `reports/publish/gemma4_ctx_text_matrix_2026-04-03.tsv`
- `reports/publish/gemma4_context_compaction_q4_2026-04-03_r2.log`
- `llama-cpp-vulkan/out/gemma4-compaction/gemma4_compaction_q4_2026-04-03-r2.json`
- `reports/publish/gemma4_mcp_wiki_paintings_2026-04-03_r10.log`
- `mcp/out/wiki_paintings_2026-04-03-r10/summary.json`

### Qwen3.5-9B Q8_0 (GGUF, VLM)

Quantized Qwen3.5 vision-language model validated here with one scripted three-task suite:
1) visual understanding (image description), 2) text summarization, 3) coding generation.

Model card: [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/qwen3.5-9b-gguf/Qwen3.5-9B-Q8_0.gguf \
      MMPROJ_PATH=$MODEL_ROOT/qwen3.5-9b-gguf/mmproj-F16.gguf \
      PORT=8131 MODEL_TAG=qwen35_9b_q8_2026_03_08b CTX_SIZE=32768 MAX_TOKENS=768 \
      GPU_LAYERS=999 MEM_LIMIT=75g MEMORY_SWAP=75g MEM_RESERVATION=67g \
      LLAMA_DEVICE=Vulkan0 REASONING_BUDGET=0 REASONING_FORMAT=none \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_qwen35_vlm_task_suite.sh
```

Evidence:
- `reports/publish/qwen35_9b_q8_task_suite_2026-03-08b.log`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_9b_q8_2026_03_08b_vision.json`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_9b_q8_2026_03_08b_summarization.json`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_9b_q8_2026_03_08b_coding.json`
- `reports/research/qwen35_9b_q8_verify.json`

Simple-text context probe (`4k, 8k, 16k, 32k, 64k, 128k`; descending with skip-after-first-success) hit `ctx=131072`.

```bash
bash $REPO_ROOT/llama-cpp-vulkan/scripts/check_qwen35_ctx_text_matrix.sh
```

Context evidence:
- `reports/publish/qwen35_ctx_text_matrix_2026-03-08.tsv`
- `reports/publish/qwen35_ctx_text_matrix_latest_2026-03-08.tsv`
- `reports/publish/qwen35_ctx_text_max_2026-03-08.tsv`
- `reports/publish/qwen35_ctx_text/qwen35_9b_q8_ctx131072.log`
- `llama-cpp-vulkan/out/qwen35-ctx-text/qwen35_9b_q8_ctx131072_2026-03-08.json`

### Qwen3.5-27B Q8_0 (GGUF, VLM)

Same three-task suite as above, with higher-capacity quantized weights.

Model card: [`unsloth/Qwen3.5-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/qwen3.5-27b-gguf/Qwen3.5-27B-Q8_0.gguf \
      MMPROJ_PATH=$MODEL_ROOT/qwen3.5-27b-gguf/mmproj-F16.gguf \
      PORT=8132 MODEL_TAG=qwen35_27b_q8_2026_03_08b CTX_SIZE=16384 MAX_TOKENS=768 \
      GPU_LAYERS=999 MEM_LIMIT=75g MEMORY_SWAP=75g MEM_RESERVATION=67g \
      LLAMA_DEVICE=Vulkan0 REASONING_BUDGET=0 REASONING_FORMAT=none \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_qwen35_vlm_task_suite.sh
```

Evidence:
- `reports/publish/qwen35_27b_q8_task_suite_2026-03-08b.log`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_27b_q8_2026_03_08b_vision.json`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_27b_q8_2026_03_08b_summarization.json`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_27b_q8_2026_03_08b_coding.json`
- `reports/research/qwen35_27b_q8_verify.json`

Simple-text context probe max: `ctx=131072`.

Context evidence:
- `reports/publish/qwen35_ctx_text/qwen35_27b_q8_ctx131072.log`
- `llama-cpp-vulkan/out/qwen35-ctx-text/qwen35_27b_q8_ctx131072_2026-03-08.json`

### Qwen3.5-122B-A10B Q4_K_M (GGUF, VLM)

Mixture-of-experts Qwen3.5 variant validated with the same three-task suite.  
On this host/profile, this run is documented in CPU fallback mode (`GPU_LAYERS=0`) for stability.

Model card: [`unsloth/Qwen3.5-122B-A10B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/qwen3.5-122b-a10b-gguf/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
      MMPROJ_PATH=$MODEL_ROOT/qwen3.5-122b-a10b-gguf/mmproj-F16.gguf \
      PORT=8133 MODEL_TAG=qwen35_122b_a10b_q4km_2026_03_08b CTX_SIZE=8192 MAX_TOKENS=512 \
      GPU_LAYERS=0 MEM_LIMIT=85g MEMORY_SWAP=85g MEM_RESERVATION=76g \
      LLAMA_DEVICE=Vulkan0 REASONING_BUDGET=0 REASONING_FORMAT=none \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_qwen35_vlm_task_suite.sh
```

Evidence:
- `reports/publish/qwen35_122b_a10b_q4km_task_suite_2026-03-08b.log`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_122b_a10b_q4km_2026_03_08b_vision.json`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_122b_a10b_q4km_2026_03_08b_summarization.json`
- `llama-cpp-vulkan/out/qwen35-task-suite/qwen35_122b_a10b_q4km_2026_03_08b_coding.json`
- `reports/research/qwen35_122b_a10b_q4km_verify.json`

Simple-text context probe max: `ctx=131072`.

Context evidence:
- `reports/publish/qwen35_ctx_text/qwen35_122b_a10b_q4km_ctx131072.log`
- `llama-cpp-vulkan/out/qwen35-ctx-text/qwen35_122b_a10b_q4km_ctx131072_2026-03-08.json`

Full-GPU confirmation (`GPU_LAYERS=999`, Vulkan, no CPU-fallback) was tested for both requested contexts and failed during model load:
- `ctx=65536`: `reports/publish/qwen35_122b_a10b_q4km_ctx65536_gpu999_min12_short_watchdog.log`
- `ctx=131072`: `reports/publish/qwen35_122b_a10b_q4km_ctx131072_gpu999_min12_short_watchdog.log`

Observed error pattern in both logs:
- `llama_params_fit: ... n_gpu_layers already set by user to 999, abort`
- `.radv/amdgpu: Not enough memory for command submission`
- `vk::Queue::submit: ErrorDeviceLost`

### Qwen3-Next-80B-A3B (GGUF)

Mixture-of-experts chat model served via llama.cpp (OpenAI-compatible endpoint). This section validates the maximum practical context size under the strict memory policy.

Model card: [`Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce (validated `ctx=196608`, `max_tokens=1024`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf \
      PORT=8113 CTX_SIZE=196608 MAX_TOKENS=1024 PROMPT_MODE=text THREADS=8 GPU_LAYERS=999 \
      OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/qwen3_next_80b_q5_ctx_196608_75g_retest2.json \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh
```

Evidence:
- `reports/publish/llama_qwen3_next_ctx196608.log`
- `llama-cpp-vulkan/out/qwen3_next_80b_q5_ctx_196608_75g_retest2.json`

### Qwen3-Coder-Next (GGUF)

Coding-focused MoE model, also served via llama.cpp.

Model card: [`Qwen/Qwen3-Coder-Next-GGUF`](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce (validated `ctx=196608`, `max_tokens=1536`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf \
      PORT=8114 CTX_SIZE=196608 MAX_TOKENS=1536 PROMPT_MODE=coding THREADS=8 GPU_LAYERS=999 \
      OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/qwen3_coder_next_q5_ctx_196608_75g_retest2.json \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh
```

Evidence:
- `reports/publish/llama_qwen3_coder_next_ctx196608.log`
- `llama-cpp-vulkan/out/qwen3_coder_next_q5_ctx_196608_75g_retest2.json`

### Qwen2.5-Coder-32B (GGUF)

Coding model capped here at its documented `128k` maximum context.

Model card: [`Qwen/Qwen2.5-Coder-32B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce (validated `ctx=131072`, `max_tokens=1536`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/qwen2.5-coder-32b-instruct-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf \
      PORT=8116 CTX_SIZE=131072 MAX_TOKENS=1536 PROMPT_MODE=coding THREADS=8 GPU_LAYERS=999 \
      OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/qwen25_coder_32b_q4_ctx_131072_75g_retest2.json \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh
```

Evidence:
- `reports/publish/llama_qwen25_coder_32b_ctx131072.log`
- `llama-cpp-vulkan/out/qwen25_coder_32b_q4_ctx_131072_75g_retest2.json`

### GPT-OSS-120B MXFP4 (GGUF)

Large general-chat model tested via llama.cpp with `MXFP4` shards.

Model card: [`ggml-org/gpt-oss-120b-GGUF`](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)
Docker image: `llama-cpp-vulkan:latest` (build: see [Containers](#containers)).

Reproduce (validated `ctx=131072`, `max_tokens=1024`):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env MODEL_PATH=$MODEL_ROOT/gpt-oss-120b-gguf/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
      PORT=8115 CTX_SIZE=131072 MAX_TOKENS=1024 PROMPT_MODE=text THREADS=8 GPU_LAYERS=999 \
      OUT_JSON=$REPO_ROOT/llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_ctx_131072_75g_retest2.json \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/probe_ctx_once.sh
```

Evidence:
- `reports/publish/llama_gpt_oss_120b_ctx131072.log`
- `llama-cpp-vulkan/out/gpt_oss_120b_mxfp4_ctx_131072_75g_retest2.json`

### VS Code (Continue) integration

Continue supports OpenAI-compatible endpoints. Generate a sample config:

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/write_continue_config.sh
```

Install the Continue extension in VS Code, then copy the generated output to your Continue config location (for example `~/.continue/config.yaml`).
The generated config uses `provider: openai` with `apiBase` pointing at the local llama.cpp servers, and a dummy `apiKey`. Update ports or model names as needed.

Reference: https://docs.continue.dev/customize/model-setup/openai

## MCP

These are the MCP workflow demos requested. Outputs are kept in `mcp/out/` and linked here for auditability.

Compliant automation note: this repository does **not** include CAPTCHA solving or instructions for bypassing anti-bot protections. The Playwright demo stops if it detects a CAPTCHA/interstitial instead of attempting evasion.

### Gemma 4 Playwright + Docker shell tool demo

This run validates MCP tool usage with Gemma 4 in a constrained/safe way:
- Gemma 4 picks famous paintings from a curated Wikipedia candidate set.
- Playwright MCP navigates each selected page and saves markdown + screenshot evidence.
- Gemma 4 proposes a shell-tool resize plan in JSON.
- The shell command is executed by MCP, but only as `docker run ... mcp-image-tools:1.0` on a mounted output directory under `mcp/out/`, not on the host shell.

Script + containers:
- Orchestrator: `mcp/scripts/run_gemma4_mcp_wiki_paintings_demo.sh`
- Model server image: `llama-cpp-vulkan:latest`
- Shell tool image: `mcp-image-tools:1.0` (built by `mcp/scripts/build_image_tools.sh`)

Reproduce (validated run `2026-04-03-r10`):

```bash
$REPO_ROOT/scripts/run_with_activity_watchdog.sh \
  --log $REPO_ROOT/reports/publish/gemma4_mcp_wiki_paintings_2026-04-03_r10.log \
  --max-seconds 3600 --idle-seconds 900 --poll-seconds 20 \
  --watch-path $REPO_ROOT/mcp/out/wiki_paintings_2026-04-03-r10 \
  -- $REPO_ROOT/scripts/run_test_with_cleanup.sh \
    env OUT_TAG=2026-04-03-r10 PLAYWRIGHT_PORT=8937 SHELL_MCP_PORT=8025 LLM_PORT=8155 \
        MODEL_PATH=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf \
        MMPROJ_PATH=$MODEL_ROOT/gemma4-26b-a4b-it-gguf/mmproj-gemma-4-26B-A4B-it-f16.gguf \
        RESIZE_IMAGE_TAG=mcp-image-tools:1.0 \
    bash $REPO_ROOT/mcp/scripts/run_gemma4_mcp_wiki_paintings_demo.sh
```

Outcome:
- Run finished `cmd_status=0`.
- Downloaded and processed four pages: `Mona Lisa`, `The Starry Night`, `The Persistence of Memory`, `Girl with a Pearl Earring`.
- Docker resize + identify + montage steps all returned `exit_code=0`.
- Final outputs are bounded by `<=1024`, but source images are Wikipedia thumbnails (`330px` wide), so no upscaling is applied by design.

Evidence:
- `reports/publish/gemma4_mcp_wiki_paintings_2026-04-03_r10.log`
- `mcp/out/wiki_paintings_2026-04-03-r10/summary.json`
- `mcp/out/wiki_paintings_2026-04-03-r10/llm_plans.json`
- `mcp/out/wiki_paintings_2026-04-03-r10/paintings_contact_sheet.png`

![Gemma MCP paintings contact sheet](mcp/out/wiki_paintings_2026-04-03-r10/paintings_contact_sheet.png)
Caption: `Gemma 4 + MCP` | Playwright Wikipedia browse + dockerized shell resizing | `4` paintings | `max_side<=1024`

### Playwright BBC scrape

Script: `mcp/scripts/playwright_bbc_scrape.py`

Evidence:
- `reports/publish/mcp_playwright_bbc_scrape.log`
- `mcp/out/bbc_2026-02-15/bbc_news.md` + `mcp/out/bbc_2026-02-15/bbc_news.png`

### Playwright reliability demo

Script: `mcp/scripts/playwright_reliability_demo.py`

Evidence:
- `reports/publish/mcp_playwright_reliability_demo.log`
- `mcp/out/bbc_reliable_2026-02-15/summary.json`

### Excel calc demo

Script: `mcp/scripts/excel_calc_demo.py`

Evidence:
- `reports/publish/mcp_excel_calc_demo.log`
- `mcp/out/excel_2026-02-15/demo.xlsx` + `mcp/out/excel_2026-02-15/summary.json`

### Python and shell demo

Script: `mcp/scripts/python_shell_demo.py`

Evidence:
- `reports/publish/mcp_python_shell_demo.log`
- `mcp/out/python_shell_2026-02-15/summary.json`

## Audio

### Kokoro-82M (TTS)

Long-form text-to-speech generation (podcast-style transcript chunking).

Model card: [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)
Docker image: `local-audio-tools:latest` (build: `audio/scripts/build_audio_tools.sh`).

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env OUT_WAV=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav \
      OUT_SUMMARY=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_summary.json \
  bash $REPO_ROOT/audio/scripts/test_kokoro_podcast.sh
```

Evidence:
- `reports/publish/audio_kokoro_tts.log`
- `audio/out/podcast_kokoro_best_retest.wav`
- `audio/out/podcast_kokoro_best_retest_summary.json`

### faster-whisper (subtitles)

Speech-to-text subtitle extraction to `.srt` from a known audio file.

Model card: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
Docker image: `local-audio-tools:latest` (build: `audio/scripts/build_audio_tools.sh`).

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env INPUT_AUDIO=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav \
      OUT_SRT=$REPO_ROOT/audio/out/podcast_kokoro_best_retest.srt \
      OUT_TXT=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_transcript.txt \
      OUT_SUMMARY=$REPO_ROOT/audio/out/podcast_kokoro_best_retest_stt_summary.json \
  bash $REPO_ROOT/audio/scripts/test_faster_whisper_subtitles.sh
```

Evidence:
- `reports/publish/audio_faster_whisper_subtitles.log`
- `audio/out/podcast_kokoro_best_retest.srt`
- `audio/out/podcast_kokoro_best_retest_transcript.txt`

### Voxtral-Mini-3B-2507 (fallback)

Practical transcription fallback validated on this host.

Model card: [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
Docker image: `voxtral-rocm:latest` (build: `audio/scripts/build_voxtral_tools.sh`).

Reproduce (validated `30s` clip):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  env CLIP_SECONDS=30 \
  bash $REPO_ROOT/audio/scripts/test_voxtral_mini_3b_2507_transcribe.sh
```

Evidence:
- `reports/publish/audio_voxtral_mini_3b_transcribe.log`
- `audio/out/voxtral_mini_3b_2507_transcript.txt`

## 3D Reconstruction

### VGGT-1B

Multi-view 3D reconstruction from public real-place photos, with a browser point-cloud viewer.

Model card: [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B)
Docker image: `vggt-rocm:latest` (build: `reconstruction-3d/scripts/build_vggt_image.sh`).

Reproduce (12 photos -> point cloud + preview):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/reconstruction-3d/scripts/build_vggt_image.sh
$REPO_ROOT/reconstruction-3d/scripts/download_south_building_views.sh
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/reconstruction-3d/scripts/run_vggt_reconstruct.sh
```

Viewer:
- Start local server: `PORT=8093 bash reconstruction-3d/scripts/serve_viewer.sh`
- Open: `http://127.0.0.1:8093/reconstruction-3d/viewer/`

Evidence:
- `reports/publish/vggt_reconstruct.log`
- `reconstruction-3d/out/south_building/south_building_points.ply`
- `reconstruction-3d/out/south_building/south_building_points_preview.png`

![VGGT South Building point cloud preview](reconstruction-3d/out/south_building/south_building_points_preview.png)
Caption: `VGGT-1B` | 12 views -> `200k` point cloud | `load_resolution=768`, `model_resolution=448`

## Video

### Wan2.1-T2V-1.3B

Local text-to-video generation (short clip).

Model card: [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
Docker image: uses `stable-diffusion-rocm:latest` (see `video/README.md`).

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/video/scripts/test_wan21_t2v_sample.sh
```

Evidence:
- `reports/publish/wan21_t2v.log`
- `video/out/wan21_t2v_sample.mp4`
- `video/out/wan21_t2v_sample_frame0.png`

![Wan2.1-T2V first frame](video/out/wan21_t2v_sample_frame0.png)
Caption: `Wan2.1-T2V-1.3B` | `672x384`, `17` frames, `fps=8`, `steps=8`

## LLM Quantization and Fine-Tuning Demos

### Quantization demo (GGUF)

This demo converts a small FP16 GGUF into Q4, then runs an inference smoke test.

Model: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
Tools container: `llama-cpp-tools:latest` (build: `llm-quantize/Dockerfile`)

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/llm-quantize/scripts/quantize_qwen25_05b_fp16_to_q4km.sh
```

Evidence:
- `reports/publish/llm_quantize_demo.log`
- `reports/quantize/quantize_outputs.json`
- `reports/quantize/fp16_response.txt`
- `reports/quantize/q4_response.txt`

### Fine-tuning demo (CPU LoRA)

Tiny LoRA SFT run on CPU (intentionally small dataset + short training) plus a base-vs-tuned comparison.

Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
Fine-tune container: `llm-finetune:latest` (build: `llm-finetune/Dockerfile`)

Reproduce:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/llm-finetune/scripts/finetune_smollm2_135m_lora_demo.sh
```

Evidence:
- `reports/publish/llm_finetune_demo.log`
- `reports/finetune/finetune_comparison.json`

## Agentic Coding Experiments

### Aider and local coder model demo

This demo shows an “agentic” workflow: Aider edits a small C# repo while talking to a local coder model served by llama.cpp (OpenAI-compatible endpoint).

Docs: `agentic/dotnet-demo/README.md`

Docker images:
- Model server: `llama-cpp-vulkan:latest`
- Demo app (built by the run): `dotnet-sum-api:local`

Reproduce:

1) Start the coder model server (terminal 1, long-running; watchdog disabled):

```bash
RUN_WITH_WATCHDOG=0 $REPO_ROOT/scripts/run_memsafe.sh \
  env PORT=8004 CTX_SIZE=196608 GPU_LAYERS=999 THREADS=8 \
      MODEL=$MODEL_ROOT/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_server.sh
```

2) Install Aider (host):

```bash
$REPO_ROOT/agentic/scripts/install_aider.sh
```

3) Run Aider against the local model (terminal 2):

```bash
$REPO_ROOT/agentic/scripts/run_aider_host.sh
```

Alternative (containerized Aider):

```bash
$REPO_ROOT/agentic/scripts/run_aider_container.sh
```

4) Build and run the generated API container:

```bash
$REPO_ROOT/agentic/scripts/build_dotnet_container.sh
```

Artifacts:
- `reports/publish/agentic_dotnet_demo_build_and_run.log`
- `agentic/out/dotnet_build.log`
- `agentic/out/dotnet_run_response.json`

### Staged GA optimizer demo (with MCP)

Staged implementation of a multithreaded genetic optimizer for `double[]` in C# (tests-first progression + internal plan + MCP-assisted benchmark preparation).

README: `agentic/ga-optimizer-demo/README.md`

Artifacts:
- `reports/publish/agentic_ga_optimizer_tests.log`
- `agentic/ga-optimizer-demo/out/tests_post_impl_recheck_publish.log`

## Tested Environment (Actual Values)

These are the **exact values** used during the tests documented in this repo:

- **OS:** Fedora Linux 43 (Workstation Edition)
- **Kernel:** `6.18.8-200.fc43.x86_64`
- **Kernel cmdline (tested):** `iommu=pt amdgpu.gttsize=90112 ttm.pages_limit=23068672 ttm.page_pool_size=23068672`
- **Model storage:** `$MODEL_ROOT` (tested value: `/mnt/hf/models`)
- **HF cache root:** `$HF_ROOT` (tested value: `/mnt/hf`, via `HF_HOME=$HF_ROOT`)
- **Workspace:** `$REPO_ROOT` (tested in a non-root home-directory clone)

If you need to replicate exactly, keep the same kernel and command line, and use the same model paths.

## Host Setup (Fedora 43 / Strix Halo)

These steps tune UMA behavior and device permissions so ROCm/Vulkan workloads are stable on early Strix Halo hardware.
They are intentionally conservative and should be applied gradually, with a reboot and verification after each change.

### Kernel parameters (96GB UMA baseline)

```bash
sudo grubby --update-kernel=ALL --args="iommu=pt amdgpu.gttsize=73728 ttm.pages_limit=18874368 ttm.page_pool_size=18874368"
# Optional if you need more UMA headroom:
# sudo grubby --update-kernel=ALL --args="amdttm.pages_limit=18874368 amdttm.page_pool_size=18874368"
```

Optional higher UMA target (only if stable and you need more headroom):

```bash
sudo grubby --update-kernel=ALL --args="iommu=pt amdgpu.gttsize=90112 ttm.pages_limit=23068672 ttm.page_pool_size=23068672"
```

For the tests in this repository, the **higher UMA target** above (gttsize 90112 / pages 23068672) was used.

### Verify after reboot

```bash
cat /proc/cmdline
lsmod | grep amdgpu
ls -l /dev/kfd /dev/dri
```

### User permissions

```bash
sudo usermod -aG video,render $USER
newgrp render
```

### Kernel stability checks

```bash
dmesg -T | grep -iE "amdgpu|kfd|sva|iommu|ttm" | tail -n 200
```

Notes:
- Avoid the reported 6.18.3-200 regression for Strix Halo; prefer a known-good adjacent 6.18.x build.
- If you see SVA/IOMMU bind errors, reduce UMA settings or switch kernels.

### Docker baseline

```bash
sudo dnf install -y moby-engine docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Enable BuildKit:

```bash
sudo mkdir -p /etc/docker
cat <<'EOF' | sudo tee /etc/docker/daemon.json
{
  "features": { "buildkit": true }
}
EOF
sudo systemctl restart docker
```

## Initial Machine Configuration (Fedora 43 + Kernel 6.18)

This section captures the original host setup steps used to install and pin a 6.18 kernel on Fedora 43 (from prior local session notes).
It is kept here so you can reproduce the exact kernel baseline used during validation.

### Install kernel 6.18 from updates-testing (kernel-only)

```bash
rpm -q kernel
sudo dnf list kernel --enablerepo=updates-testing
sudo dnf install \
  kernel kernel-core kernel-modules kernel-modules-extra \
  --enablerepo=updates-testing \
  --setopt=install_weak_deps=False
rpm -q kernel
```

### Pin the default kernel (grubby + BLS)

```bash
sudo grubby --info=ALL | grep -E "kernel=|title="
sudo grubby --set-default /boot/vmlinuz-6.18.x-*.fc43.x86_64
sudo grubby --default-kernel
```

If Fedora keeps booting the last-used kernel, boot the desired 6.18 entry once from the menu, then verify:

```bash
uname -r
cat /proc/cmdline
```

To force the default permanently:

```bash
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' /etc/default/grub
sudo sed -i 's/^GRUB_SAVEDEFAULT=.*/GRUB_SAVEDEFAULT=false/' /etc/default/grub
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
```

### Lock kernel packages (prevent accidental upgrades)

```bash
sudo dnf install -y dnf-plugins-core
sudo dnf versionlock add kernel kernel-core kernel-modules kernel-modules-extra
dnf versionlock list
```

### Apply kernel parameters to the active 6.18 kernel

```bash
sudo grubby --update-kernel=/boot/vmlinuz-6.18.x-*.fc43.x86_64 \
  --args="iommu=pt amdgpu.gttsize=73728 ttm.pages_limit=18874368 ttm.page_pool_size=18874368"
cat /proc/cmdline
```

Notes:
- Avoid the reported 6.18.3-200 regression on Strix Halo. Prefer a nearby known-good 6.18.x build.
- If you need more UMA headroom later, increase gttsize/ttm pages gradually and stress-test.

## Containers

Each container has a dedicated README with build + usage details:

- Qwen-Image: `qwen-image/README.md`
- Qwen-Image-Edit: `qwen-image-edit/README.md`
- Stable Diffusion: `stable-diffusion/README.md`
- Llama.cpp Vulkan: `llama-cpp-vulkan/README.md`
- Qwen-VL (Vision): `qwen-vl/README.md`
- Vision Detection + Pose: `vision-detection/README.md`
- Video generation workflows: `video/README.md` (uses `stable-diffusion-rocm:latest`)

Quick build commands:

```bash
cd $REPO_ROOT/qwen-image && DOCKER_BUILDKIT=1 docker build -f Dockerfile.qwen-image.rocm -t qwen-image-rocm:latest .
cd $REPO_ROOT/qwen-image-edit && DOCKER_BUILDKIT=1 docker build -t qwen-image-edit-rocm:latest .
cd $REPO_ROOT/stable-diffusion && DOCKER_BUILDKIT=1 docker build -t stable-diffusion-rocm:latest .
cd $REPO_ROOT/llama-cpp-vulkan && DOCKER_BUILDKIT=1 docker build -t llama-cpp-vulkan:latest .
cd $REPO_ROOT/qwen-vl && DOCKER_BUILDKIT=1 docker build -f Dockerfile.qwen-vl.rocm -t qwen-vl-rocm:latest .
cd $REPO_ROOT/vision-detection && DOCKER_BUILDKIT=1 docker build -t vision-yolo-rocm:latest .
```

### Source Versions (Build-Time Notes)

Some images build from upstream source without a pinned commit:
- `stable-diffusion`: installs `diffusers` from the current `main` branch at build time (to support Flux pipelines).
- `llama-cpp-vulkan`: builds `llama.cpp` from the default branch at build time.

If you need reproducibility, pin the commit/branch in the Dockerfiles and rebuild.

## Model Downloads

Model weights are stored in `$MODEL_ROOT` to keep containers stateless and make re-runs fast.
The download scripts all resume partial files, which is important on slow or unstable connections.

Vision-language model:

```bash
$REPO_ROOT/qwen-vl/scripts/download_qwen_vl_7b.sh
```

Full Qwen-Image models (large downloads; optional):

```bash
$REPO_ROOT/qwen-image/scripts/download_qwen_image.sh
$REPO_ROOT/qwen-image/scripts/download_qwen_image_2512.sh
$REPO_ROOT/qwen-image-edit/scripts/download_qwen_image_edit.sh
$REPO_ROOT/qwen-image-edit/scripts/download_qwen_image_edit_2511.sh
$REPO_ROOT/qwen-image-edit/scripts/download_qwen_image_edit_2509.sh
```

Text-only LLMs (GGUF):

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen3_next_80b.sh
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen3_coder_next.sh
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen3_32b_q6.sh
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen3_coder_30b_q8.sh
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen25_coder_32b.sh
```

Qwen3.5 VLM quantized bundle (requested `9B 8-bit`, `27B 8-bit`, `122B-A10B 4-bit`), with checksum verification:

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/download_qwen35_vlm_quant_bundle.sh
```

The script downloads these GGUF + mmproj pairs:
- `unsloth/Qwen3.5-9B-GGUF` (`Qwen3.5-9B-Q8_0.gguf` + `mmproj-F16.gguf`)
- `unsloth/Qwen3.5-27B-GGUF` (`Qwen3.5-27B-Q8_0.gguf` + `mmproj-F16.gguf`)
- `unsloth/Qwen3.5-122B-A10B-GGUF` (`Q4_K_M` shard set + `mmproj-F16.gguf`)
These are pre-quantized GGUF files from HF; there is no local quantization/conversion step in this flow.

Verification reports are written under `reports/research/`:
- `reports/research/qwen35_9b_q8_verify.json`
- `reports/research/qwen35_27b_q8_verify.json`
- `reports/research/qwen35_122b_a10b_q4km_verify.json`

After download, run the task matrix (`visual understanding`, `text summarization`, `coding`) for all three:

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/run_qwen35_quant_task_matrix.sh
```

Default runtime profile in the matrix script:
- `9B Q8_0`: `ctx=32768`, `n_gpu_layers=999`, `mem=75g`
- `27B Q8_0`: `ctx=16384`, `n_gpu_layers=999`, `mem=75g`
- `122B-A10B Q4_K_M`: `ctx=8192`, `n_gpu_layers=0` (CPU-fallback for stability), `mem=85g`
These can be overridden via environment variables (e.g., `CTX_122B`, `GPU_LAYERS_122B`, `MEM_LIMIT_122B`).

Separate simple-text context sweep (`4k..128k`, descending) is scripted in:
- `llama-cpp-vulkan/scripts/check_qwen35_ctx_text_matrix.sh`
Latest run evidence:
- `reports/publish/qwen35_ctx_text_max_2026-03-08.tsv`

Container compatibility note:
- No new Docker image is required for these quantized runs; they are intended for the existing `llama-cpp-vulkan:latest` container.
- Visual understanding uses GGUF + `mmproj` (`--mmproj` passed via `EXTRA_ARGS`).
- If parsing/loading fails on a stale image, rebuild `llama-cpp-vulkan:latest` from current `master` (this repo Dockerfile builds `llama.cpp` from source).

Bigger diffusion model (SDXL base):

```bash
$REPO_ROOT/stable-diffusion/scripts/download_sdxl_base.sh
```

Stable Diffusion 3.5 models:

```bash
$REPO_ROOT/stable-diffusion/scripts/download_sd35_medium.sh
$REPO_ROOT/stable-diffusion/scripts/download_sd35_large.sh
```

Flux model (modern diffusion baseline):

```bash
$REPO_ROOT/stable-diffusion/scripts/download_flux2_klein_4b.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_klein_9b.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_dev.sh
$REPO_ROOT/stable-diffusion/scripts/download_flux2_dev_nvfp4.sh
```

Diffusion upscaler model:

```bash
$REPO_ROOT/stable-diffusion/scripts/download_sd_x4_upscaler.sh
```

Video generation model:

```bash
$REPO_ROOT/video/scripts/download_wan21_t2v_1_3b.sh
```

GPT-OSS-120B (GGUF MXFP4 shards):

```bash
$REPO_ROOT/llama-cpp-vulkan/scripts/download_gpt_oss_120b.sh
```

Notes:
- Large downloads (SDXL, Qwen3-Next 80B, Qwen2.5-Coder 32B) can take hours on slow links.
- Very large download: GPT-OSS-120B GGUF MXFP4 is split into three files, total about 60 GB.
- Re-running the scripts resumes partial downloads under `$MODEL_ROOT`.

## Cleanup Policy

This repo includes a small test harness used for the publish-day reruns to keep the host stable and avoid stale containers/processes skewing results. The same harness is useful for normal day-to-day runs as a “memory-safe wrapper” around any script.

Cleanup scripts:
- `scripts/cleanup_machine.sh` — kills non-essential high-RSS processes (default threshold `5GB`) and removes inference/test containers; writes logs under `$REPO_ROOT/reports/cleanup/`.
- `scripts/run_test_with_cleanup.sh` — wrapper that runs cleanup before and after any command.
- `scripts/run_memsafe.sh` — reusable wrapper for production-style runs; exports memory-safe defaults, then calls `run_test_with_cleanup.sh`.
- `scripts/preflight_memory_guard.sh` — aborts a run when host headroom/swap usage is unsafe.

Container memory policy:
- All test/server scripts default to `MEM_LIMIT=75g`, `MEMORY_SWAP=75g`, and `MEM_RESERVATION=67g` (`--memory` + `--memory-swap` + `--memory-reservation`).
- Use `scripts/run_memsafe.sh` for repeatable safe execution outside this test suite.

Swap safety (host):
- Current swap on this machine:
  - zram: `/dev/zram0` (`8G`)
  - swapfile: `/var/swap/strix-halo.swap` (`64G`, `pri=10`)
- The swapfile is created/enabled/persisted by:
  - `scripts/setup_swapfile.sh` (defaults: `SWAP_PATH=/var/swap/strix-halo.swap`, `SWAP_SIZE_GB=64`, `SWAP_PRIO=10`)
- Verify swap at any time:

```bash
free -h
swapon --show
```

Example usage (run a single model test under the harness):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/qwen-image/scripts/test_qwen_image.sh
```

Reusable pattern for any long-running model server:

```bash
MEM_LIMIT=75g MEMORY_SWAP=75g MEM_RESERVATION=67g \
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/llama-cpp-vulkan/scripts/run_server.sh
```

`run_memsafe.sh` behavior (what it does internally):
- Loads `scripts/env.sh` (`REPO_ROOT`, `HF_ROOT`, `MODEL_ROOT`).
- Exports memory policy defaults (`MEM_LIMIT`, `MEMORY_SWAP`, `MEM_RESERVATION`, `OOM_SCORE_ADJ`).
- Runs the command under an activity watchdog (`scripts/run_with_activity_watchdog.sh`) with bounded defaults:
  - `RUN_MAX_SECONDS=3600` (hard stop)
  - `RUN_IDLE_SECONDS=600` (kills runs that go quiet)
  - Host safety guard during the run (prevents memory-deadlock freezes):
    - `MIN_AVAILABLE_GB=16` (kills if host `MemAvailable` drops below this)
    - `MAX_SWAP_USED_GB=24` (kills if host `SwapUsed` exceeds this)
- Runs `cleanup_machine.sh` and `preflight_memory_guard.sh` before and after the command.

Manual cleanup (same cleanup pass the harness runs pre/post):

```bash
$REPO_ROOT/scripts/cleanup_machine.sh
```

## Passwordless sudo and Non-root copy

If you prefer non-root operation, enable passwordless sudo for your local user and keep your working copy under your home directory.
This keeps root-only experiments isolated while making repeat runs easier.

Enable NOPASSWD for your user (one time, replace `<user>`):

```bash
sudo tee /etc/sudoers.d/<user> <<'EOF'
<user> ALL=(ALL) NOPASSWD:ALL
EOF
sudo chmod 440 /etc/sudoers.d/<user>
```

Copy a root-owned lab repo to your user workspace and fix ownership:

```bash
export REPO_ROOT="$HOME/strix-halo-inference"
sudo rsync -a --delete /path/to/root-copy/ "$REPO_ROOT/"
sudo chown -R "$USER:$USER" "$REPO_ROOT"
```

In the non-root copy, run scripts with `sudo` where they call Docker or modify system state.
Model paths are controlled by `scripts/env.sh`; override `HF_ROOT` once and `MODEL_ROOT` follows automatically.
