FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    "kokoro>=0.9.2" \
    "soundfile>=0.12" \
    "numpy>=1.26,<3" \
    "faster-whisper>=1.1.0"

WORKDIR /workspace
