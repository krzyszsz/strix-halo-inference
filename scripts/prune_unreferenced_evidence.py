#!/usr/bin/env python3
"""
Prune old/unreferenced evidence artifacts to keep the repo publish-friendly.

Policy (default):
- Keep evidence files that are referenced from *documentation* files:
  - all `README.md` files (repo root + subfolders)
- Consider evidence candidates as tracked files under:
  - `reports/**`
  - `docs/images/**`
  - any `*/out/**`

This intentionally ignores PLAN.md so historical working notes don't force keeping
large piles of old artifacts.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

REPO_REL_PREFIXES = (
    "reports/",
    "docs/",
    "agentic/",
    "audio/",
    "llama-cpp-vulkan/",
    "mcp/",
    "qwen-image/",
    "qwen-image-edit/",
    "qwen-vl/",
    "reconstruction-3d/",
    "stable-diffusion/",
    "video/",
    "vision-detection/",
    "vision-face/",
    "llm-quantize/",
    "llm-finetune/",
)


def _git_ls_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], text=True, cwd=REPO_ROOT).splitlines()


def _is_evidence(path: str) -> bool:
    return path.startswith("reports/") or path.startswith("docs/images/") or "/out/" in path


def _repo_rel(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return None


def _extract_paths_from_md(doc: Path, md: str) -> set[str]:
    out: set[str] = set()
    base = doc.parent

    def add_raw(raw: str) -> None:
        raw = raw.strip()
        if not raw:
            return
        if raw.startswith(("http://", "https://", "data:", "#")):
            return
        # Strip common wrappers/punctuation.
        raw = raw.strip().strip("()[]{}<>")
        raw = raw.rstrip(".,;:")
        raw = raw.replace("\\", "/")

        # Expand $REPO_ROOT prefix if present.
        if raw.startswith("$REPO_ROOT/"):
            raw2 = raw[len("$REPO_ROOT/") :]
            p = (REPO_ROOT / raw2).resolve()
            rel = _repo_rel(p)
            if rel and _is_evidence(rel):
                out.add(rel)
            return
        # Sometimes `$REPO_ROOT/...` appears in prose and gets matched without `$`.
        if raw.startswith("REPO_ROOT/"):
            raw2 = raw[len("REPO_ROOT/") :]
            p = (REPO_ROOT / raw2).resolve()
            rel = _repo_rel(p)
            if rel and _is_evidence(rel):
                out.add(rel)
            return

        # Ignore absolute paths.
        if raw.startswith(("/", "~")):
            return

        # Repo-relative paths (most docs use this style).
        if raw.startswith(REPO_REL_PREFIXES):
            p = (REPO_ROOT / raw).resolve()
            rel = _repo_rel(p)
            if rel and _is_evidence(rel):
                out.add(rel)
            return

        # Resolve doc-relative paths like `out/foo.png`.
        p = (base / raw).resolve()
        rel = _repo_rel(p)
        if rel and _is_evidence(rel):
            out.add(rel)

    # Markdown images/links: ![alt](path) and [text](path)
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        add_raw(m.group(1))
    for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", md):
        add_raw(m.group(1))

    # Inline code spans: `...`
    for m in re.finditer(r"`([^`]+)`", md):
        raw = m.group(1).strip()
        if not raw:
            continue
        # Avoid flag/env snippets; we mostly care about file-like tokens.
        if any(ch in raw for ch in (" ", "=")):
            continue
        # Keep only path-looking tokens.
        if "/" not in raw:
            continue
        add_raw(raw)

    # Raw tokens (unbackticked) that look like repo paths.
    # This catches e.g. "reports/foo.log," in prose.
    raw_token_re = re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)")
    for m in raw_token_re.finditer(md):
        add_raw(m.group(1))

    return out


def _extract_doc_references() -> set[str]:
    refs: set[str] = set()
    docs = sorted([p for p in REPO_ROOT.rglob("README.md") if ".git/" not in str(p)])
    for doc in docs:
        md = doc.read_text(encoding="utf-8", errors="replace")
        refs |= _extract_paths_from_md(doc, md)
    return refs


def _chunked(xs: list[str], n: int) -> list[list[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually delete files (git rm).")
    ap.add_argument("--keep-missing", action="store_true", help="Also keep paths referenced but missing on disk.")
    args = ap.parse_args()

    tracked = _git_ls_files()
    candidates = sorted([p for p in tracked if _is_evidence(p)])
    refs = _extract_doc_references()

    keep: set[str] = set()
    missing: list[str] = []
    for p in sorted(refs):
        if (REPO_ROOT / p).exists():
            keep.add(p)
        else:
            missing.append(p)
            if args.keep_missing:
                keep.add(p)

    unref = [p for p in candidates if p not in keep]

    print(f"Docs scanned: {len(list(REPO_ROOT.rglob('README.md')))} README.md files")
    print(f"Evidence candidates (tracked): {len(candidates)}")
    print(f"Referenced paths (resolved): {len(refs)}")
    print(f"Referenced existing: {len(keep)}")
    if missing:
        print(f"Referenced but missing on disk: {len(missing)}")
        for s in missing[:20]:
            print(f"  - {s}")
    print(f"Unreferenced evidence to prune: {len(unref)}")

    if not unref:
        return 0

    # Report largest deletions to make the output actionable.
    sizes: list[tuple[int, str]] = []
    for p in unref:
        try:
            sizes.append(((REPO_ROOT / p).stat().st_size, p))
        except FileNotFoundError:
            pass
    sizes.sort(reverse=True)
    print("Largest unreferenced evidence files:")
    for sz, p in sizes[:30]:
        print(f"  - {sz/1024/1024:6.2f} MiB  {p}")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")
        return 0

    # Delete with git rm in small batches (avoid arg length limits).
    for chunk in _chunked(unref, 200):
        subprocess.check_call(["git", "rm", "-f", "--", *chunk], cwd=REPO_ROOT)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
