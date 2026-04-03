#!/usr/bin/env python3
"""
Build a simple README collage image from a fixed set of evidence outputs.

This keeps the repo's "hero" image reproducible without requiring ImageMagick.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a grid collage from images.")
    p.add_argument("--out", required=True, help="Output PNG path (repo-relative or absolute).")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--tile", type=int, default=700, help="Tile size in pixels (square).")
    p.add_argument("--bg", default="#0b0f14", help="Background color for letterboxing.")
    p.add_argument("images", nargs="+", help="Input image paths (in grid order).")
    return p.parse_args()


def _tile(im: Image.Image, tile: int, bg: str) -> Image.Image:
    im = im.convert("RGB")
    fitted = ImageOps.contain(im, (tile, tile), method=Image.LANCZOS)
    canvas = Image.new("RGB", (tile, tile), color=bg)
    x = (tile - fitted.size[0]) // 2
    y = (tile - fitted.size[1]) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def main() -> int:
    args = _parse_args()
    cols = args.cols
    rows = args.rows
    tile = args.tile

    expected = cols * rows
    if len(args.images) != expected:
        raise SystemExit(f"Expected exactly {expected} images ({cols}x{rows}), got {len(args.images)}.")

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collage = Image.new("RGB", (cols * tile, rows * tile), color=args.bg)

    for idx, img_path in enumerate(args.images):
        p = Path(img_path).expanduser()
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            p = repo_root / p
        if not p.exists():
            raise SystemExit(f"Missing input image: {p}")

        r = idx // cols
        c = idx % cols
        im = Image.open(p)
        collage.paste(_tile(im, tile=tile, bg=args.bg), (c * tile, r * tile))

    collage.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

