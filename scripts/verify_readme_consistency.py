#!/usr/bin/env python3
"""
Quick publish-time sanity checks for README.md:
- Contents anchors exist (GitHub-style heading slug approximation).
- Referenced local artifact paths exist.

This is intentionally lightweight (no extra dependencies).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"
REPO_ROOT = README.parent

PATH_PREFIXES = (
    "reports/",
    "docs/",
    "scripts/",
    "qwen-image/",
    "qwen-image-edit/",
    "stable-diffusion/",
    "qwen-vl/",
    "vision-detection/",
    "vision-face/",
    "llama-cpp-vulkan/",
    "agentic/",
    "audio/",
    "video/",
    "mcp/",
    "reconstruction-3d/",
    "llm-quantize/",
    "llm-finetune/",
)


def gh_anchor(text: str, used: dict[str, int]) -> str:
    # Approximate GitHub heading slug:
    # - Lowercase
    # - Convert whitespace to '-' first
    # - Strip anything except [a-z0-9-]
    # This preserves double-hyphens in cases like "A / B" -> "a--b" (GitHub behavior).
    base = text.strip().lower()
    base = re.sub(r"\s+", "-", base)
    base = re.sub(r"[^a-z0-9\-]", "", base).strip("-")
    if not base:
        base = "section"

    n = used.get(base, 0)
    used[base] = n + 1
    return base if n == 0 else f"{base}-{n}"


def iter_headings(md: str) -> list[tuple[int, str, str]]:
    used: dict[str, int] = {}
    out: list[tuple[int, str, str]] = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        # Skip code-fence headings (rare) and empty headings.
        if not title or title.startswith("```"):
            continue
        anchor = gh_anchor(title, used)
        out.append((level, title, anchor))
    return out


def extract_contents_anchors(md: str) -> list[str]:
    # Extract anchors within the Contents section only.
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "## Contents":
            start = i + 1
            break
    if start is None:
        return []

    end = None
    for i in range(start, len(lines)):
        if re.match(r"^##\s+", lines[i]):
            end = i
            break
    if end is None:
        end = len(lines)

    anchors: list[str] = []
    for line in lines[start:end]:
        for m in re.finditer(r"\(#([a-z0-9\-]+)\)", line):
            anchors.append(m.group(1))
    return anchors


def extract_local_paths(md: str) -> set[Path]:
    paths: set[Path] = set()

    # Markdown images: ![alt](path)
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        raw = m.group(1).strip()
        if raw.startswith(("http://", "https://")):
            continue
        if raw.startswith("#") or raw.startswith("data:"):
            continue
        if not raw.startswith(PATH_PREFIXES):
            continue
        paths.add((REPO_ROOT / raw).resolve())

    # Evidence-style bullets: - `path`, `path2`
    for line in md.splitlines():
        if not line.lstrip().startswith("- "):
            continue
        for m in re.finditer(r"`([^`]+)`", line):
            raw = m.group(1).strip()
            if not raw or raw.startswith(("http://", "https://")):
                continue
            if raw.startswith("#") or raw.startswith("$"):
                continue
            if any(ch in raw for ch in " ="):
                continue
            # Only validate obvious relative paths (avoid licenses/ids/etc).
            if "/" not in raw:
                continue
            # Ignore obvious non-path tokens.
            if raw.startswith(("bash", "python", "sudo", "env")):
                continue
            if raw.startswith(("/", "~")):
                # README should not point at absolute paths as evidence.
                continue
            if not raw.startswith(PATH_PREFIXES):
                continue
            paths.add((REPO_ROOT / raw).resolve())

    return paths


def main() -> int:
    if not README.exists():
        print(f"ERROR: missing {README}", file=sys.stderr)
        return 2

    md = README.read_text(encoding="utf-8")

    headings = iter_headings(md)
    anchor_set = {a for _, _, a in headings}
    toc_anchors = extract_contents_anchors(md)

    missing_anchors = [a for a in toc_anchors if a not in anchor_set]
    if missing_anchors:
        print("ERROR: Contents anchors missing from headings:")
        for a in missing_anchors:
            print(f"  - #{a}")
    else:
        print(f"OK: Contents anchors ({len(toc_anchors)}) all exist")

    local_paths = extract_local_paths(md)
    missing_paths: list[str] = []
    for p in sorted(local_paths):
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if not p.exists():
            missing_paths.append(str(p.relative_to(REPO_ROOT)))

    if missing_paths:
        print("ERROR: Missing referenced paths:")
        for s in missing_paths:
            print(f"  - {s}")
    else:
        print(f"OK: Referenced local paths exist ({len(local_paths)})")

    return 0 if (not missing_anchors and not missing_paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
