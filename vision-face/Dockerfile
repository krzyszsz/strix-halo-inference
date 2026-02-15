ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install -U \
    numpy \
    onnxruntime \
    opencv-python-headless \
    pillow

WORKDIR /app
COPY scripts/infer_face_match.py /app/infer_face_match.py

ENTRYPOINT ["python", "/app/infer_face_match.py"]

