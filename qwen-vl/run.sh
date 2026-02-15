#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8005}

exec uvicorn app:app --host "$HOST" --port "$PORT"
