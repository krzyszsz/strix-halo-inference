#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RAW_DIR="$REPO_ROOT/reconstruction-3d/data/eiffel_tower/images_raw"
IMG_DIR="$REPO_ROOT/reconstruction-3d/data/eiffel_tower/images"
SRC_TSV="$REPO_ROOT/reconstruction-3d/data/eiffel_tower/sources.tsv"

mkdir -p "$RAW_DIR" "$IMG_DIR"

cat > "$SRC_TSV" <<'TSV'
local_name	source_url
view_01.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Eiffel%20Tower%20from%20north%20Avenue%20de%20New%20York%2C%20Aug%202010.jpg
view_02.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Tour%20eiffel%20at%20sunrise%20from%20the%20trocadero.jpg
view_03.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Eiffel%20Tower%20Paris%20June%202010.jpg
view_04.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Eiffel%20Tower%20from%20northwest%2C%20August%202010.jpg
view_05.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Eiffel%20Tower%20as%20seen%20from%20the%20Pont%20Mirabeau%2C%2022%20April%202014.jpg
view_06.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Tour%20Eiffel%20-%2020150801%2015h30%20%2810621%29.jpg
view_07.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Tour%20Eiffel%20-%2020150801%2013h44%20%2810613%29.jpg
view_08.jpg	https://commons.wikimedia.org/wiki/Special:FilePath/Tour%20Eiffel%2C%20Paris%20%28IMG%2020211231%20152254%29.jpg
TSV

while IFS=$'\t' read -r local_name source_url; do
  [ "$local_name" = "local_name" ] && continue
  raw_path="$RAW_DIR/$local_name"
  if [ ! -f "$raw_path" ]; then
    echo "Downloading $local_name"
    curl -L --fail --retry 3 -o "$raw_path" "$source_url"
  fi
done < "$SRC_TSV"

REPO_ROOT_PY="$REPO_ROOT" python - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageOps

repo = Path(os.environ["REPO_ROOT_PY"])
raw_dir = repo / "reconstruction-3d/data/eiffel_tower/images_raw"
out_dir = repo / "reconstruction-3d/data/eiffel_tower/images"
out_dir.mkdir(parents=True, exist_ok=True)

max_side = 1600
for raw_path in sorted(raw_dir.glob("view_*.jpg")):
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
