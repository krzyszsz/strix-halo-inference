#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$REPO_ROOT/reconstruction-3d/data/south_building"
RAW_DIR="$DATA_DIR/images_raw"
IMG_DIR="$DATA_DIR/images"
SRC_TSV="$DATA_DIR/sources.tsv"
ZIP_PATH="$DATA_DIR/south-building.zip"
ZIP_URL="https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
KEEP_ZIP="${KEEP_ZIP:-0}"

mkdir -p "$DATA_DIR" "$RAW_DIR" "$IMG_DIR"

if [ ! -f "$ZIP_PATH" ]; then
  echo "Downloading south-building dataset archive"
  curl -L --fail --retry 3 -o "$ZIP_PATH" "$ZIP_URL"
fi

# Contiguous burst from the same camera/date to keep geometry simple and lighting consistent.
IMAGE_LIST=(
  P1180168.JPG
  P1180169.JPG
  P1180170.JPG
  P1180171.JPG
  P1180172.JPG
  P1180173.JPG
  P1180174.JPG
  P1180175.JPG
  P1180176.JPG
  P1180177.JPG
  P1180178.JPG
  P1180179.JPG
)

{
  printf "local_name\tsource_url\n"
  for name in "${IMAGE_LIST[@]}"; do
    printf "%s\t%s\n" "$name" "$ZIP_URL"
  done
} > "$SRC_TSV"

for name in "${IMAGE_LIST[@]}"; do
  out="$RAW_DIR/$name"
  if [ ! -f "$out" ]; then
    unzip -oj "$ZIP_PATH" "south-building/images/$name" -d "$RAW_DIR"
  fi
done

REPO_ROOT_PY="$REPO_ROOT" python - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageOps

repo = Path(os.environ["REPO_ROOT_PY"])
raw_dir = repo / "reconstruction-3d/data/south_building/images_raw"
out_dir = repo / "reconstruction-3d/data/south_building/images"
out_dir.mkdir(parents=True, exist_ok=True)

max_side = 1600
for raw_path in sorted(raw_dir.glob("P118*.JPG")):
    img = Image.open(raw_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
      img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    out_path = out_dir / raw_path.name
    img.save(out_path, format="JPEG", quality=95)
    print(out_path)
PY

if [ "$KEEP_ZIP" != "1" ]; then
  rm -f "$ZIP_PATH"
fi
