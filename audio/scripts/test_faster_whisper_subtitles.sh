#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

docker() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

IMAGE="${IMAGE:-local-audio-tools:latest}"
INPUT_AUDIO="${INPUT_AUDIO:-$REPO_ROOT/audio/out/podcast_kokoro_best_retest.wav}"
MODEL_SIZE="${MODEL_SIZE:-small}"
DEVICE="${DEVICE:-cpu}"
COMPUTE_TYPE="${COMPUTE_TYPE:-int8}"
LANGUAGE="${LANGUAGE:-en}"
OUT_DIR="$REPO_ROOT/audio/out"
OUT_SRT="${OUT_SRT:-$OUT_DIR/podcast_kokoro_best_retest.srt}"
OUT_TXT="${OUT_TXT:-$OUT_DIR/podcast_kokoro_best_retest_transcript.txt}"
OUT_SUMMARY="${OUT_SUMMARY:-$OUT_DIR/podcast_kokoro_best_retest_stt_summary.json}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

mkdir -p "$OUT_DIR"

if [ ! -f "$INPUT_AUDIO" ]; then
  echo "Missing input audio: $INPUT_AUDIO" >&2
  exit 1
fi

TMP_PY="$(mktemp "${TMPDIR:-/tmp}/whisper_srt.XXXXXX.py")"
trap 'rm -f "$TMP_PY"' EXIT

cat <<'PY' > "$TMP_PY"
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel


def fmt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


input_audio = Path(os.environ["INPUT_AUDIO"])
model_size = os.environ.get("MODEL_SIZE", "small")
device = os.environ.get("DEVICE", "cpu")
compute_type = os.environ.get("COMPUTE_TYPE", "int8")
language = os.environ.get("LANGUAGE", "en")
out_srt = Path(os.environ["OUT_SRT"])
out_txt = Path(os.environ["OUT_TXT"])
out_summary = Path(os.environ["OUT_SUMMARY"])

model = WhisperModel(model_size, device=device, compute_type=compute_type)
segments, info = model.transcribe(
    str(input_audio),
    language=language,
    vad_filter=True,
)
segments = list(segments)

srt_lines = []
txt_lines = []
for idx, seg in enumerate(segments, start=1):
    srt_lines.append(str(idx))
    srt_lines.append(f"{fmt_time(seg.start)} --> {fmt_time(seg.end)}")
    text = seg.text.strip()
    srt_lines.append(text)
    srt_lines.append("")
    txt_lines.append(text)

out_srt.parent.mkdir(parents=True, exist_ok=True)
out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

duration = getattr(info, "duration", None)
summary = {
    "input_audio": str(input_audio),
    "model_size": model_size,
    "device": device,
    "compute_type": compute_type,
    "language": language,
    "segments": len(segments),
    "duration_seconds": None if duration is None else round(float(duration), 2),
    "srt_path": str(out_srt),
    "transcript_path": str(out_txt),
}
out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

docker run --rm \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  -v "$REPO_ROOT:$REPO_ROOT:Z" \
  -v "$TMP_PY:/tmp/whisper_srt.py:Z" \
  -w "$REPO_ROOT" \
  -e INPUT_AUDIO="$INPUT_AUDIO" \
  -e MODEL_SIZE="$MODEL_SIZE" \
  -e DEVICE="$DEVICE" \
  -e COMPUTE_TYPE="$COMPUTE_TYPE" \
  -e LANGUAGE="$LANGUAGE" \
  -e OUT_SRT="$OUT_SRT" \
  -e OUT_TXT="$OUT_TXT" \
  -e OUT_SUMMARY="$OUT_SUMMARY" \
  --entrypoint python \
  "$IMAGE" \
  -u /tmp/whisper_srt.py

for file in "$OUT_SRT" "$OUT_TXT" "$OUT_SUMMARY"; do
  if [ -f "$file" ] && [ ! -w "$file" ]; then
    sudo chown "$(id -u):$(id -g)" "$file" || true
  fi
done
