#!/usr/bin/env python3
"""
Compact a publish summary TSV that may have duplicate rows for the same test name.

We keep:
- the original header (first line)
- the last occurrence of each `name` (first column)
- the order of first appearance of each name
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to summary TSV (in-place).")
    args = ap.parse_args()

    p = Path(args.path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return 0

    header = lines[0].rstrip("\n")
    order: list[str] = []
    last: dict[str, str] = {}

    for line in lines[1:]:
        if not line.strip():
            continue
        name = line.split("\t", 1)[0]
        if name not in last:
            order.append(name)
        last[name] = line

    out_lines = [header] + [last[n] for n in order if n in last]
    p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

