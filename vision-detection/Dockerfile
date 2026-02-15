ARG BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install -U ultralytics opencv-python-headless

WORKDIR /app
COPY scripts/infer_yolo.py /app/infer_yolo.py

ENTRYPOINT ["python", "/app/infer_yolo.py"]
