#!/usr/bin/env python
import json
import os
import re
import socket
import subprocess
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

from mcp_sse_client import McpSseClient

BASE_URL = "https://www.bbc.co.uk/"
PLAYWRIGHT_PORT = int(os.getenv("PLAYWRIGHT_PORT", "8931"))
LLM_BASE = os.getenv("LLM_BASE", "http://127.0.0.1:8004/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-gguf")
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "mcp"


def wait_port(port: int, timeout: int = 60) -> str | None:
    start = time.time()
    while time.time() - start < timeout:
        for host in ("::1", "127.0.0.1"):
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            with socket.socket(family) as s:
                s.settimeout(1)
                try:
                    s.connect((host, port))
                    return host
                except OSError:
                    continue
        time.sleep(1)
    return None


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path)
    return slug.strip("-") or "page"


def llm_select_links(candidates: list[str], max_links: int = 6) -> list[str]:
    try:
        resp = requests.get(f"{LLM_BASE}/models", timeout=3)
        if resp.status_code != 200:
            return candidates[:max_links]
    except Exception:
        return candidates[:max_links]

    prompt = (
        "Select up to {n} BBC subpage URLs from the list below. "
        "Return ONLY a JSON array of URLs, no extra text.\n\n".format(n=max_links)
        + "\n".join(candidates)
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise JSON generator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 256,
    }
    try:
        r = requests.post(f"{LLM_BASE}/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            chosen = json.loads(match.group(0))
            if isinstance(chosen, list):
                filtered = [u for u in chosen if u in candidates]
                return filtered[:max_links] if filtered else candidates[:max_links]
    except Exception:
        return candidates[:max_links]
    return candidates[:max_links]


def main() -> None:
    today = date.today().isoformat()
    out_dir = MCP_ROOT / "out" / f"bbc_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "playwright-mcp.log"

    proc = subprocess.Popen(
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
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        host = wait_port(PLAYWRIGHT_PORT, timeout=90)
        if not host:
            raise RuntimeError("Playwright MCP server did not start")

        base_host = f"[{host}]" if ":" in host else host
        client = McpSseClient(f"http://{base_host}:{PLAYWRIGHT_PORT}")
        client.initialize()
        # Ensure browser is installed (no args)
        client.call_tool("browser_install")

        client.call_tool("browser_navigate", {"url": BASE_URL})
        client.call_tool("browser_wait_for", {"time": 2})

        js = (
            "async (page) => {"
            "const links = Array.from(document.querySelectorAll('a'))"
            ".map(a => a.href).filter(h => h && h.startsWith('https://www.bbc.co.uk/'));"
            "const uniq = Array.from(new Set(links));"
            "const top = uniq.filter(u => {"
            "  const p = new URL(u).pathname.split('/').filter(Boolean);"
            "  return p.length === 1;"
            "});"
            "return top.slice(0, 12);"
            "}"
        )
        result = client.call_tool("browser_run_code", {"code": js})
        candidates = result if isinstance(result, list) else []
        if not candidates:
            candidates = [
                "https://www.bbc.co.uk/news",
                "https://www.bbc.co.uk/sport",
                "https://www.bbc.co.uk/weather",
                "https://www.bbc.co.uk/reel",
                "https://www.bbc.co.uk/culture",
            ]

        selected = llm_select_links(candidates, max_links=6)

        captured = []
        for url in selected:
            slug = slugify(url)
            md_name = f"mcp/out/bbc_{today}/bbc_{slug}.md"
            png_name = f"mcp/out/bbc_{today}/bbc_{slug}.png"

            client.call_tool("browser_navigate", {"url": url})
            client.call_tool("browser_wait_for", {"time": 2})
            client.call_tool("browser_snapshot", {"filename": md_name})
            client.call_tool("browser_take_screenshot", {"filename": png_name})

            captured.append({
                "url": url,
                "markdown": md_name,
                "screenshot": png_name,
            })

        summary = {
            "date": today,
            "base_url": BASE_URL,
            "output_dir": str(out_dir),
            "pages": captured,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
