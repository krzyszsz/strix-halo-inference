FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Keep versions pinned for reproducibility.
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
 && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu \
 && python -m pip install --no-cache-dir \
    transformers==4.57.6 \
    peft==0.13.2 \
    accelerate==1.12.0 \
    safetensors==0.7.0 \
    huggingface_hub==0.36.2

ENTRYPOINT ["/bin/bash", "-lc"]
