#!/usr/bin/env python3
import json
import os
import re
import shlex
import socket
import subprocess
import time
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse

import requests

from mcp_sse_client import McpSseClient

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "mcp"

LLM_BASE = os.getenv("LLM_BASE", "http://127.0.0.1:8153/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-gguf")
PLAYWRIGHT_PORT = int(os.getenv("PLAYWRIGHT_PORT", "8935"))
SHELL_MCP_PORT = int(os.getenv("SHELL_MCP_PORT", "8023"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "4"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))
RESIZE_IMAGE_TAG = os.getenv("RESIZE_IMAGE_TAG", "mcp-image-tools:1.0")
DOCKER_TIMEOUT = int(os.getenv("DOCKER_TIMEOUT", "900"))
OUT_TAG = os.getenv("OUT_TAG", date.today().isoformat())
BASE_URL = os.getenv("WIKI_BASE_URL", "https://en.wikipedia.org/wiki/List_of_most_expensive_paintings")
HOST_UID = os.getuid()
HOST_GID = os.getgid()


def wait_port(port: int, timeout: int = 90) -> str | None:
    started = time.time()
    while time.time() - started < timeout:
        for host in ("::1", "127.0.0.1"):
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                try:
                    sock.connect((host, port))
                    return host
                except OSError:
                    continue
        time.sleep(1)
    return None


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return slug or "item"


def parse_first_json(text: str):
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except Exception:
            continue
    return None


def unwrap_tool_result(result):
    if isinstance(result, (list, str, int, float, bool)):
        return result
    if not isinstance(result, dict):
        return None
    if "structuredContent" in result:
        return result["structuredContent"]
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                parsed = parse_first_json(text)
                return parsed if parsed is not None else text
    return result


def llm_json(prompt: str, max_tokens: int = 512):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{LLM_BASE}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    obj = parse_first_json(text)
    return text, obj


def normalize_wiki_image_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def widen_wiki_thumbnail(url: str, target_px: int) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if "/thumb/" not in path:
        return url
    path2 = re.sub(r"/\d+px-([^/]+)$", rf"/{target_px}px-\1", path)
    if path2 == path:
        return url
    return urlunparse((parsed.scheme, parsed.netloc, path2, "", "", ""))


def infer_ext(url: str, content_type: str) -> str:
    ctype = (content_type or "").lower()
    if "png" in ctype:
        return ".png"
    if "webp" in ctype:
        return ".webp"
    if "jpeg" in ctype or "jpg" in ctype:
        return ".jpg"
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def download_file(url: str, out_path: Path) -> dict:
    headers = {"User-Agent": "strix-halo-inference/1.0 (Gemma4 MCP demo)"}
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return {
            "status_code": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "bytes": out_path.stat().st_size,
        }


def wiki_summary_image(page_url: str) -> dict:
    parsed = urlparse(page_url)
    title = parsed.path.rsplit("/", 1)[-1]
    title = unquote(title).replace(" ", "_")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    headers = {"User-Agent": "strix-halo-inference/1.0 (Gemma4 MCP demo)"}
    r = requests.get(api_url, timeout=60, headers=headers)
    r.raise_for_status()
    data = r.json()
    # Prefer thumbnail first to keep demo downloads bounded and reproducible.
    img = ""
    if isinstance(data.get("thumbnail"), dict):
        img = str(data["thumbnail"].get("source", "")).strip()
    if not img and isinstance(data.get("originalimage"), dict):
        img = str(data["originalimage"].get("source", "")).strip()
    return {"title": data.get("title", title), "image_url": img, "api_url": api_url}


def build_resize_cmd(host_dir: Path, max_side: int) -> str:
    script = (
        "set -euo pipefail; "
        f"mkdir -p /work/resized_{max_side}; "
        "shopt -s nullglob; "
        "for f in /work/raw/*; do "
        "  [ -f \"$f\" ] || continue; "
        f"  convert \"$f\" -auto-orient -resize '{max_side}x{max_side}>' "
        f"\"/work/resized_{max_side}/$(basename \"$f\")\"; "
        "done; "
        f"ls -1 /work/resized_{max_side}"
    )
    return (
        f"docker run --rm --security-opt label=disable --user {HOST_UID}:{HOST_GID} "
        f"-v {shlex.quote(str(host_dir))}:/work "
        f"{shlex.quote(RESIZE_IMAGE_TAG)} /bin/bash -lc {shlex.quote(script)}"
    )


def build_identify_cmd(host_dir: Path, max_side: int) -> str:
    script = (
        "set -euo pipefail; "
        f"identify -format '%f %w %h\\n' /work/resized_{max_side}/*"
    )
    return (
        f"docker run --rm --security-opt label=disable --user {HOST_UID}:{HOST_GID} "
        f"-v {shlex.quote(str(host_dir))}:/work "
        f"{shlex.quote(RESIZE_IMAGE_TAG)} /bin/bash -lc {shlex.quote(script)}"
    )


def build_montage_cmd(host_dir: Path, max_side: int) -> str:
    script = (
        "set -euo pipefail; "
        "shopt -s nullglob; "
        f"files=(/work/resized_{max_side}/*); "
        "[ ${#files[@]} -gt 0 ]; "
        "montage \"${files[@]}\" "
        "-thumbnail '256x256^' -gravity center -extent 256x256 "
        "-tile 2x -geometry +8+8 -background '#0b1220' "
        "/work/paintings_contact_sheet.png"
    )
    return (
        f"docker run --rm --security-opt label=disable --user {HOST_UID}:{HOST_GID} "
        f"-v {shlex.quote(str(host_dir))}:/work "
        f"{shlex.quote(RESIZE_IMAGE_TAG)} /bin/bash -lc {shlex.quote(script)}"
    )


def parse_identify(stdout: str) -> list[dict]:
    rows: list[dict] = []
    for line in (stdout or "").splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        try:
            rows.append({"file": parts[0], "width": int(parts[1]), "height": int(parts[2])})
        except ValueError:
            continue
    return rows


def call_browser_run_code(client: McpSseClient, code: str):
    raw = client.call_tool("browser_run_code", {"code": code})
    return unwrap_tool_result(raw), raw


def main() -> None:
    out_dir = MCP_ROOT / "out" / f"wiki_paintings_{OUT_TAG}"
    raw_dir = out_dir / "raw"
    resized_dir = out_dir / f"resized_{MAX_SIDE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    pw_log = out_dir / "playwright-mcp.log"
    plan_json = out_dir / "llm_plans.json"
    summary_json = out_dir / "summary.json"

    proc_pw = subprocess.Popen(
        [
            "npx",
            "@playwright/mcp@latest",
            "--port",
            str(PLAYWRIGHT_PORT),
            "--allowed-hosts",
            "*",
            "--output-dir",
            str(out_dir),
            "--output-mode",
            "file",
            "--no-sandbox",
        ],
        stdout=pw_log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    env_shell = os.environ.copy()
    env_shell.update({"MCP_TRANSPORT": "sse", "MCP_PORT": str(SHELL_MCP_PORT)})
    proc_shell = subprocess.Popen(
        ["python3", str(MCP_ROOT / "servers" / "local_shell_mcp.py")],
        env=env_shell,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    summary = {
        "date": date.today().isoformat(),
        "out_dir": str(out_dir),
        "base_url": BASE_URL,
        "llm_base": LLM_BASE,
        "llm_model": LLM_MODEL,
        "selected_pages": [],
        "downloaded_files": [],
        "tool_calls": [],
    }
    llm_plans = {}

    try:
        pw_host = wait_port(PLAYWRIGHT_PORT, timeout=120)
        if not pw_host:
            raise RuntimeError("Playwright MCP server did not start")
        shell_host = wait_port(SHELL_MCP_PORT, timeout=60)
        if not shell_host:
            raise RuntimeError("Shell MCP server did not start")

        pw_client_host = f"[{pw_host}]" if ":" in pw_host else pw_host
        sh_client_host = f"[{shell_host}]" if ":" in shell_host else shell_host
        pw_client = McpSseClient(f"http://{pw_client_host}:{PLAYWRIGHT_PORT}")
        sh_client = McpSseClient(f"http://{sh_client_host}:{SHELL_MCP_PORT}")
        pw_client.initialize()
        sh_client.initialize()

        pw_client.call_tool("browser_install")
        pw_client.call_tool("browser_navigate", {"url": BASE_URL})
        pw_client.call_tool("browser_wait_for", {"time": 2})

        candidates = [
            {"title": "Mona Lisa", "url": "https://en.wikipedia.org/wiki/Mona_Lisa"},
            {"title": "The Starry Night", "url": "https://en.wikipedia.org/wiki/The_Starry_Night"},
            {"title": "The Persistence of Memory", "url": "https://en.wikipedia.org/wiki/The_Persistence_of_Memory"},
            {"title": "Girl with a Pearl Earring", "url": "https://en.wikipedia.org/wiki/Girl_with_a_Pearl_Earring"},
            {"title": "The Night Watch", "url": "https://en.wikipedia.org/wiki/The_Night_Watch"},
            {"title": "The Scream", "url": "https://en.wikipedia.org/wiki/The_Scream"},
            {"title": "Guernica", "url": "https://en.wikipedia.org/wiki/Guernica_(Picasso)"},
            {"title": "The Last Supper", "url": "https://en.wikipedia.org/wiki/The_Last_Supper_(Leonardo)"},
        ]
        summary["tool_calls"].append(
            {
                "tool": "candidate_list(static_curated)",
                "count": len(candidates),
                "source_page": BASE_URL,
            }
        )

        dedup = {}
        for row in candidates:
            dedup[row["url"]] = row
        candidate_list = list(dedup.values())[:40]

        candidate_lines = [f"- {row['title']} | {row['url']}" for row in candidate_list]
        selection_prompt = (
            f"/no_think Choose up to {MAX_PAGES} famous paintings from the candidate list.\n"
            "Return strict JSON only in this format:\n"
            "{\"pages\":[{\"title\":\"...\",\"url\":\"...\"}]}\n"
            "Only URLs from the candidate list are allowed.\n\n"
            "Candidates:\n"
            + "\n".join(candidate_lines)
        )
        raw_text, parsed_selection = llm_json(selection_prompt, max_tokens=512)
        llm_plans["page_selection"] = {"raw": raw_text, "parsed": parsed_selection}

        selected = []
        if isinstance(parsed_selection, dict) and isinstance(parsed_selection.get("pages"), list):
            allowed = {row["url"]: row["title"] for row in candidate_list}
            for page in parsed_selection["pages"]:
                if not isinstance(page, dict):
                    continue
                url = str(page.get("url", "")).strip()
                title = str(page.get("title", "")).strip() or allowed.get(url, "")
                if url in allowed:
                    selected.append({"title": title or allowed[url], "url": url})
        if not selected:
            selected = candidate_list[:MAX_PAGES]
        selected = selected[:MAX_PAGES]
        summary["selected_pages"] = selected

        for idx, page in enumerate(selected, start=1):
            url = page["url"]
            slug = slugify(page["title"])
            md_name = f"mcp/out/wiki_paintings_{OUT_TAG}/wiki_{idx:02d}_{slug}.md"
            png_name = f"mcp/out/wiki_paintings_{OUT_TAG}/wiki_{idx:02d}_{slug}.png"

            pw_client.call_tool("browser_navigate", {"url": url})
            pw_client.call_tool("browser_wait_for", {"time": 2})
            pw_client.call_tool("browser_snapshot", {"filename": md_name})
            pw_client.call_tool("browser_take_screenshot", {"filename": png_name})

            summary["tool_calls"].append(
                {
                    "tool": "browser_navigate+snapshot+screenshot",
                    "page_url": url,
                    "snapshot_md": md_name,
                    "snapshot_png": png_name,
                }
            )

            image_meta = wiki_summary_image(url)
            image_url = normalize_wiki_image_url(str(image_meta.get("image_url", "")).strip())
            preferred_url = widen_wiki_thumbnail(image_url, MAX_SIDE)
            if not image_url:
                continue

            head = requests.head(preferred_url, timeout=30, allow_redirects=True)
            ext = infer_ext(preferred_url, head.headers.get("content-type", ""))
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            out_name = f"{idx:02d}_{slug}{ext}"
            out_path = raw_dir / out_name
            used_url = preferred_url
            try:
                dl_meta = download_file(used_url, out_path)
            except requests.HTTPError:
                used_url = image_url
                dl_meta = download_file(used_url, out_path)

            summary["downloaded_files"].append(
                {
                    "title": page["title"],
                    "page_url": url,
                    "wiki_summary_api": image_meta.get("api_url", ""),
                    "image_url": used_url,
                    "raw_file": str(out_path),
                    "snapshot_md": md_name,
                    "snapshot_png": png_name,
                    "bytes": dl_meta["bytes"],
                    "content_type": dl_meta["content_type"],
                }
            )

        if not summary["downloaded_files"]:
            raise RuntimeError("No painting images were downloaded from selected Wikipedia pages")

        resize_plan_prompt = (
            "/no_think We need to resize all files from /work/raw to max side 1024 using the run_shell tool. "
            "The shell command must run docker and write files to /work/resized_1024. "
            "Return strict JSON only in this format: "
            "{\"tool\":\"run_shell\",\"args\":{\"operation\":\"resize_images\",\"max_side\":1024,\"timeout\":900}}"
        )
        resize_raw, resize_parsed = llm_json(resize_plan_prompt, max_tokens=200)
        llm_plans["resize_plan"] = {"raw": resize_raw, "parsed": resize_parsed}
        plan_json.write_text(json.dumps(llm_plans, indent=2), encoding="utf-8")

        resize_cmd = build_resize_cmd(out_dir, MAX_SIDE)
        tool_choice = "run_shell"
        timeout = DOCKER_TIMEOUT
        if isinstance(resize_parsed, dict):
            tool_choice = str(resize_parsed.get("tool", "run_shell")).strip() or "run_shell"
            args = resize_parsed.get("args", {})
            if isinstance(args, dict):
                try:
                    timeout = int(args.get("timeout", DOCKER_TIMEOUT))
                except Exception:
                    timeout = DOCKER_TIMEOUT

        if tool_choice != "run_shell":
            raise RuntimeError(f"LLM picked unsupported tool: {tool_choice}")

        resize_result_raw = sh_client.call_tool("run_shell", {"cmd": resize_cmd, "timeout": timeout})
        resize_result = unwrap_tool_result(resize_result_raw)
        summary["tool_calls"].append(
            {
                "tool": "run_shell(resize_in_docker)",
                "cmd": resize_cmd,
                "result": resize_result,
            }
        )
        if isinstance(resize_result, dict) and int(resize_result.get("exit_code", 1)) != 0:
            raise RuntimeError(f"resize failed: {resize_result}")

        identify_cmd = build_identify_cmd(out_dir, MAX_SIDE)
        identify_result_raw = sh_client.call_tool("run_shell", {"cmd": identify_cmd, "timeout": timeout})
        identify_result = unwrap_tool_result(identify_result_raw)
        summary["tool_calls"].append(
            {
                "tool": "run_shell(identify_dimensions_in_docker)",
                "cmd": identify_cmd,
                "result": identify_result,
            }
        )

        identify_stdout = ""
        if isinstance(identify_result, dict):
            identify_stdout = str(identify_result.get("stdout", ""))
            if int(identify_result.get("exit_code", 1)) != 0:
                raise RuntimeError(f"identify failed: {identify_result}")
        resized_dims = parse_identify(identify_stdout)
        if not resized_dims:
            raise RuntimeError("No resized files found after docker resize command")

        for row in resized_dims:
            if row["width"] > MAX_SIDE or row["height"] > MAX_SIDE:
                raise RuntimeError(f"Resize constraint violated: {row}")

        montage_cmd = build_montage_cmd(out_dir, MAX_SIDE)
        montage_result_raw = sh_client.call_tool("run_shell", {"cmd": montage_cmd, "timeout": timeout})
        montage_result = unwrap_tool_result(montage_result_raw)
        summary["tool_calls"].append(
            {
                "tool": "run_shell(create_contact_sheet_in_docker)",
                "cmd": montage_cmd,
                "result": montage_result,
            }
        )

        contact_sheet = out_dir / "paintings_contact_sheet.png"
        if not contact_sheet.exists():
            raise RuntimeError("Contact sheet was not created")

        summary["resized_dir"] = str(resized_dir)
        summary["resized_dimensions"] = resized_dims
        summary["contact_sheet"] = str(contact_sheet)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary_json)
    except Exception as exc:
        summary["error"] = str(exc)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise
    finally:
        try:
            proc_pw.terminate()
            proc_pw.wait(timeout=10)
        except Exception:
            proc_pw.kill()
        try:
            proc_shell.terminate()
            proc_shell.wait(timeout=10)
        except Exception:
            proc_shell.kill()


if __name__ == "__main__":
    main()
