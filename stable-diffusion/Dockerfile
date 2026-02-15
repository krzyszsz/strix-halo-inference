ARG ROCM_BASE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${ROCM_BASE}
ARG BNB_GIT_REF=main
ARG BNB_ROCM_ARCH=gfx1151

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/mnt/hf \
    TRANSFORMERS_CACHE=/mnt/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ID=/mnt/hf/models/sd35-medium \
    DTYPE=float16

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg libgl1 cmake ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install -U \
  "huggingface_hub[cli]" \
  "fastapi>=0.110" "uvicorn[standard]>=0.27" \
  pillow opencv-python
RUN python -m pip install -U \
  "transformers>=4.45.0,<5" "accelerate>=0.33.0" \
  bitsandbytes \
  safetensors sentencepiece protobuf
RUN git clone --depth 1 --branch "${BNB_GIT_REF}" https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bitsandbytes \
 && cd /tmp/bitsandbytes \
 && cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="${BNB_ROCM_ARCH}" -S . \
 && make -j"$(nproc)" \
 && python -m pip install . \
 && cd / \
 && rm -rf /tmp/bitsandbytes
RUN python -m pip install -U "diffusers @ https://github.com/huggingface/diffusers/archive/refs/heads/main.zip"

WORKDIR /app
COPY app.py /app/app.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

EXPOSE 8000
ENTRYPOINT ["/app/run.sh"]
