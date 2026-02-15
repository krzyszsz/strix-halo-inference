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

WORKDIR="${WORKDIR:-$REPO_ROOT/agentic/dotnet-demo}"
IMAGE="${IMAGE:-dotnet-sum-api:local}"
PORT="${PORT:-8080}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/agentic/out}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"

cd "$WORKDIR"
mkdir -p "$OUT_DIR"

docker build -t "$IMAGE" . 2>&1 | tee "${OUT_DIR}/dotnet_build.log"

CONTAINER="dotnet-sum-api-test"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  -p "${PORT}:8080" \
  "$IMAGE"

code=""
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/sum?x=3&y=4" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 2
done

if [ "$code" != "200" ]; then
  echo "API did not become ready on port ${PORT}." >&2
  docker logs --tail 200 "$CONTAINER" || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exit 1
fi

curl -s "http://127.0.0.1:${PORT}/sum?x=3&y=4" | tee "${OUT_DIR}/dotnet_run_response.json"
echo ""

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
