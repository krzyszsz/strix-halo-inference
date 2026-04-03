FROM ubuntu:24.04 AS build

ARG LLAMA_CPP_REF=master

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp.git
WORKDIR /opt/llama.cpp
RUN git checkout ${LLAMA_CPP_REF}

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target llama-cli llama-quantize

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 libgomp1 \
    python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 enforces PEP 668; use a venv for pip-installed tools.
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -U pip setuptools wheel \
 && /opt/venv/bin/pip install --no-cache-dir -U 'huggingface_hub[cli]==0.30.2'

ENV PATH=/opt/venv/bin:$PATH

COPY --from=build /opt/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=build /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=build /opt/llama.cpp/build/bin/libggml*.so* /usr/local/lib/
COPY --from=build /opt/llama.cpp/build/bin/libllama*.so* /usr/local/lib/
COPY --from=build /opt/llama.cpp/build/bin/libmtmd*.so* /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

ENTRYPOINT ["/bin/bash", "-lc"]
