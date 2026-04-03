FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends imagemagick ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

