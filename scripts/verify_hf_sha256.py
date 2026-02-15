#!/usr/bin/env python3
import argparse
import fnmatch
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def _sha256(path: str, buf_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _load_manifest(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list of repo entries")
    return data


def _verify_repo(repo_id: str, local_dir: str, include: List[str], optional: bool, fail_on_missing: bool) -> dict:
    api = HfApi()
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except HfHubHTTPError as exc:
        if optional:
            return {"repo_id": repo_id, "status": "optional-missing", "error": str(exc)}
        raise

    missing = 0
    mismatched = 0
    ok = 0
    skipped = 0

    for s in info.siblings:
        sha = None
        size = None
        if hasattr(s, "lfs") and s.lfs:
            sha = s.lfs.get("sha256")
            size = s.lfs.get("size")
        if not sha:
            continue

        if include:
            if not _match_any(s.rfilename, include):
                skipped += 1
                continue

        local_path = os.path.join(local_dir, s.rfilename)
        if not os.path.exists(local_path):
            missing += 1
            continue

        if size is not None:
            local_size = os.path.getsize(local_path)
            if local_size != size:
                mismatched += 1
                continue

        actual = _sha256(local_path)
        if actual.lower() != sha.lower():
            mismatched += 1
        else:
            ok += 1

    status = "ok" if mismatched == 0 and (missing == 0 or not fail_on_missing) else "failed"
    return {
        "repo_id": repo_id,
        "status": status,
        "ok": ok,
        "missing": missing,
        "mismatched": mismatched,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local HF downloads against SHA256 metadata.")
    default_manifest = Path(__file__).resolve().with_name("verify_hf_sha256.json")
    parser.add_argument("--manifest", default=str(default_manifest))
    parser.add_argument("--repo")
    parser.add_argument("--local-dir")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--strict", action="store_true", help="Fail on missing files as well as mismatches")
    args = parser.parse_args()

    entries = []
    if args.repo and args.local_dir:
        entries = [{
            "repo_id": args.repo,
            "local_dir": args.local_dir,
            "include": args.include,
            "optional": False,
        }]
    else:
        entries = _load_manifest(args.manifest)

    results = []
    fail = 0
    for entry in entries:
        repo_id = entry["repo_id"]
        local_dir = entry["local_dir"]
        include = entry.get("include") or []
        optional = bool(entry.get("optional", False))
        res = _verify_repo(repo_id, local_dir, include, optional, args.strict)
        results.append(res)
        if res.get("status") == "failed":
            fail = 1

    print(json.dumps(results, indent=2))
    return fail


if __name__ == "__main__":
    sys.exit(main())
