# Audio Workflows (TTS + Subtitles)

This folder contains local, reproducible audio workflows:

- Long-form TTS generation from transcript (`Kokoro-82M`).
- Subtitle extraction (`.srt`) from recorded/generated audio (`faster-whisper`).
- Mistral Voxtral speech recognition probes (latest + practical fallback).

All test scripts use Docker with default memory cap:

- `MEM_LIMIT=75g`
- `MEMORY_SWAP=75g`

## Build

```bash
export REPO_ROOT="$(pwd)"
source "$REPO_ROOT/scripts/env.sh"
cd "$REPO_ROOT"

bash audio/scripts/build_audio_tools.sh
```

## Long-Form TTS

Input transcript:

- `audio/input/podcast_script.txt`

Generate podcast-style narration:

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/audio/scripts/test_kokoro_podcast.sh
```

Outputs:

- `audio/out/podcast_kokoro_best_retest.wav`
- `audio/out/podcast_kokoro_best_retest_summary.json`

## Subtitle Extraction

Generate subtitles from an audio file (default: generated podcast):

```bash
$REPO_ROOT/scripts/run_memsafe.sh \
  bash $REPO_ROOT/audio/scripts/test_faster_whisper_subtitles.sh
```

Outputs:

- `audio/out/podcast_kokoro_best_retest.srt`
- `audio/out/podcast_kokoro_best_retest_transcript.txt`
- `audio/out/podcast_kokoro_best_retest_stt_summary.json`

## Mistral Voxtral (Speech Recognition)

This repo tracks two Voxtral paths:

- `mistralai/Voxtral-Mini-4B-Realtime-2602` (latest, realtime): **currently blocked** on this host stack.
- `mistralai/Voxtral-Mini-3B-2507` (fallback): **works** and produces a transcript on this machine.

### Build (Voxtral tools image)

```bash
bash audio/scripts/build_voxtral_tools.sh
```

### Voxtral Mini 3B (Working Transcription)

```bash
bash audio/scripts/test_voxtral_mini_3b_2507_transcribe.sh
```

Outputs:
- `audio/out/voxtral_mini_3b_2507_transcript.txt`
- `audio/out/voxtral_mini_3b_2507_summary.json`

### Voxtral Mini 4B Realtime (Blocked)

Two local probe styles exist:

1. `transformers` probe / transcribe attempt (fails; Transformers support is marked WIP upstream):

```bash
bash audio/scripts/setup_voxtral_mini_4b_realtime_2602_hf.sh
bash audio/scripts/test_voxtral_mini_4b_realtime_2602_transcribe.sh
```

2. vLLM startup probe (fails on this host ROCm stack with `libhsa-runtime64` segfaults):

```bash
PATCH_SITE=1 EXTRA_ARGS="--attention-backend ROCM_AITER_FA --load-format mistral --config-format mistral --tokenizer-mode mistral --trust-remote-code" \\
  bash audio/scripts/test_voxtral_mini_4b_realtime_2602_vllm.sh
```

Evidence logs are linked from the main `README.md` under:
Note: these Voxtral-4B probe scripts are kept for future investigation; they are not part of the publish-day rerun suite.
