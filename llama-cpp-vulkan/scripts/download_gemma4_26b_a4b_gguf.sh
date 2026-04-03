#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

require_cmd python3

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_BIN=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_BIN=(huggingface-cli download)
else
  HF_DOWNLOAD_BIN=()
fi

# Prefer plain HTTP download path for large files on hosts where Xet can be bursty/stall.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Optional accelerated downloader (requires hf_transfer package support).
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Public repo, but keep token support enabled for consistency with other scripts.
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
elif [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

MODEL_REPO="${MODEL_REPO:-ggml-org/gemma-4-26b-a4b-it-GGUF}"
LOCAL_DIR="${LOCAL_DIR:-$MODEL_ROOT/gemma4-26b-a4b-it-gguf}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/research}"
VERIFY_REPORT="${VERIFY_REPORT:-$REPORT_DIR/gemma4_26b_a4b_gguf_verify.json}"
DOWNLOAD_Q8="${DOWNLOAD_Q8:-1}"
DOWNLOAD_Q4="${DOWNLOAD_Q4:-1}"
DIRECT_HTTP="${DIRECT_HTTP:-0}"
USE_ARIA2="${USE_ARIA2:-1}"
ARIA2_SPLIT="${ARIA2_SPLIT:-6}"
ARIA2_MIN_SPLIT_SIZE="${ARIA2_MIN_SPLIT_SIZE:-64M}"

mkdir -p "$LOCAL_DIR" "$REPORT_DIR"

FILES=("mmproj-gemma-4-26B-A4B-it-f16.gguf")
if [ "$DOWNLOAD_Q8" = "1" ]; then
  FILES+=("gemma-4-26B-A4B-it-Q8_0.gguf")
fi
if [ "$DOWNLOAD_Q4" = "1" ]; then
  FILES+=("gemma-4-26B-A4B-it-Q4_K_M.gguf")
fi
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "Nothing to download. Set DOWNLOAD_Q8=1 and/or DOWNLOAD_Q4=1." >&2
  exit 2
fi

meta_tsv="$(mktemp "${TMPDIR:-/tmp}/gemma4_meta.XXXXXX.tsv")"
trap 'rm -f "$meta_tsv"' EXIT

REPO_ID="$MODEL_REPO" FILES_CSV="$(IFS=,; echo "${FILES[*]}")" python3 - <<'PY' > "$meta_tsv"
import os
from huggingface_hub import HfApi

repo = os.environ["REPO_ID"]
files = [x for x in os.environ["FILES_CSV"].split(",") if x]
api = HfApi()
info = api.model_info(repo, files_metadata=True)
nodes = {s.rfilename: s for s in info.siblings}
for rel in files:
    node = nodes.get(rel)
    size = getattr(node, "size", None) if node else None
    sha = None
    if node and getattr(node, "lfs", None):
        lfs = node.lfs
        if isinstance(lfs, dict):
            sha = lfs.get("sha256")
        else:
            sha = getattr(lfs, "sha256", None)
    size_text = "" if size is None else str(int(size))
    sha_text = "" if sha is None else str(sha)
    print(f"{rel}\t{size_text}\t{sha_text}")
PY

echo "Plan for $MODEL_REPO:"
total_bytes=0
while IFS=$'\t' read -r rel size sha; do
  if [ -n "$size" ]; then
    total_bytes=$((total_bytes + size))
    gib="$(python3 - <<PY
size=$size
print(f"{size / (1024**3):.2f}")
PY
)"
  else
    gib="unknown"
  fi
  if [ -n "$sha" ]; then
    short_sha="${sha:0:12}"
  else
    short_sha="n/a"
  fi
  echo "  - $rel (${gib} GiB, sha256=$short_sha...)"
done < "$meta_tsv"
if [ "$total_bytes" -gt 0 ]; then
  total_gib="$(python3 - <<PY
size=$total_bytes
print(f"{size / (1024**3):.2f}")
PY
)"
else
  total_gib="unknown"
fi
echo "  Total expected: ${total_gib} GiB"
echo "  destination: $LOCAL_DIR"

if [ "$USE_ARIA2" = "1" ] && command -v aria2c >/dev/null 2>&1; then
  for f in "${FILES[@]}"; do
    row="$(grep -F "${f}"$'\t' "$meta_tsv" || true)"
    size=""
    sha=""
    if [ -n "$row" ]; then
      size="$(printf '%s' "$row" | cut -f2)"
      sha="$(printf '%s' "$row" | cut -f3)"
    fi
    dest="$LOCAL_DIR/$f"
    dest_dir="$(dirname "$dest")"
    dest_base="$(basename "$dest")"
    mkdir -p "$dest_dir"
    url="https://huggingface.co/${MODEL_REPO}/resolve/main/${f}?download=true"
    echo "aria2 download: $url -> $dest"
    checksum_args=()
    if [ -n "$sha" ]; then
      checksum_args+=(--checksum="sha-256=$sha")
    fi
    aria2c \
      --console-log-level=notice \
      --summary-interval=15 \
      --max-connection-per-server="$ARIA2_SPLIT" \
      --split="$ARIA2_SPLIT" \
      --min-split-size="$ARIA2_MIN_SPLIT_SIZE" \
      --file-allocation=none \
      --allow-overwrite=true \
      --continue=true \
      --timeout=30 \
      --connect-timeout=30 \
      --retry-wait=5 \
      --max-tries=0 \
      "${checksum_args[@]}" \
      -d "$dest_dir" \
      -o "$dest_base" \
      "$url"
    if [ -n "$size" ]; then
      local_size="$(stat -c '%s' "$dest")"
      if [ "$local_size" != "$size" ]; then
        echo "Size mismatch after aria2 download: $dest (got=$local_size expected=$size)" >&2
        exit 1
      fi
    fi
  done
elif [ "$DIRECT_HTTP" = "1" ]; then
  require_cmd wget
  for f in "${FILES[@]}"; do
    dest="$LOCAL_DIR/$f"
    mkdir -p "$(dirname "$dest")"
    url="https://huggingface.co/${MODEL_REPO}/resolve/main/${f}?download=true"
    echo "wget download: $url -> $dest"
    wget -c --tries=0 --timeout=30 --retry-connrefused -O "$dest" "$url"
  done
else
  if [ "${#HF_DOWNLOAD_BIN[@]}" -eq 0 ]; then
    echo "No download backend available. Install aria2 or hf/huggingface-cli." >&2
    exit 1
  fi
  if [ "${HF_DOWNLOAD_BIN[0]}" = "hf" ]; then
    "${HF_DOWNLOAD_BIN[@]}" "$MODEL_REPO" "${FILES[@]}" \
      --local-dir "$LOCAL_DIR"
  else
    "${HF_DOWNLOAD_BIN[@]}" "$MODEL_REPO" "${FILES[@]}" \
      --local-dir "$LOCAL_DIR" \
      --local-dir-use-symlinks False
  fi
fi

for f in "${FILES[@]}"; do
  if [ ! -f "$LOCAL_DIR/$f" ]; then
    echo "Missing downloaded file: $LOCAL_DIR/$f" >&2
    exit 1
  fi
done

manifest="$(mktemp "${TMPDIR:-/tmp}/gemma4_verify_manifest.XXXXXX.json")"
trap 'rm -f "$manifest"' EXIT

MODEL_REPO="$MODEL_REPO" LOCAL_DIR="$LOCAL_DIR" FILES_CSV="$(IFS=,; echo "${FILES[*]}")" MANIFEST="$manifest" python3 - <<'PY'
import json
import os

payload = [{
    "repo_id": os.environ["MODEL_REPO"],
    "local_dir": os.environ["LOCAL_DIR"],
    "include": [x for x in os.environ["FILES_CSV"].split(",") if x],
    "optional": False,
}]
with open(os.environ["MANIFEST"], "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

python3 "$REPO_ROOT/scripts/verify_hf_sha256.py" --manifest "$manifest" --strict > "$VERIFY_REPORT"

echo "Done."
echo "Model files:"
for f in "${FILES[@]}"; do
  echo "  - $LOCAL_DIR/$f"
done
echo "Verification report:"
echo "  - $VERIFY_REPORT"
