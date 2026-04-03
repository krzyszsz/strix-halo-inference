#!/usr/bin/env python3
"""
Compute a `.safetensors` parameter count without loading tensor payloads.

This parses the safetensors header (JSON) and sums the product of each tensor shape.
That "count" is the total number of stored tensor elements (weights + buffers).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict


def _read_safetensors_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise RuntimeError(f"{path}: unexpected EOF reading header length")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise RuntimeError(f"{path}: unexpected EOF reading header payload")
    try:
        return json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"{path}: failed to parse safetensors header JSON: {e}") from e


def _human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3g}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.3g}M"
    if n >= 1_000:
        return f"{n/1_000:.3g}K"
    return str(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="Path to a .safetensors file")
    ap.add_argument("--out-json", type=Path, default=None, help="Write results to JSON")
    ap.add_argument(
        "--group-depth",
        type=int,
        default=2,
        help="How many dotted segments to keep when grouping tensor keys (default: 2)",
    )
    args = ap.parse_args()

    path: Path = args.path
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    if path.suffix != ".safetensors":
        raise SystemExit(f"Expected a .safetensors file: {path}")

    header = _read_safetensors_header(path)

    total = 0
    tensors = 0
    grouped: Dict[str, int] = {}

    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if not isinstance(shape, list):
            continue
        # Product of dimensions (supports scalars too).
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n
        tensors += 1

        parts = key.split(".")
        group = ".".join(parts[: max(1, int(args.group_depth))])
        grouped[group] = grouped.get(group, 0) + n

    result = {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "tensor_count": tensors,
        "total_elements": total,
        "total_elements_human": _human_params(total),
        "total_elements_billion": total / 1_000_000_000,
        "group_depth": int(args.group_depth),
        "grouped_elements_top10": sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

