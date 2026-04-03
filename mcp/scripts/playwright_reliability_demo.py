#!/usr/bin/env python
import json
import os
import random
import socket
import subprocess
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from mcp_sse_client import McpSseClient


BASE_URL = os.getenv("BASE_URL", "https://www.bbc.co.uk/")
PLAYWRIGHT_PORT = int(os.getenv("PLAYWRIGHT_PORT", "8932"))
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "mcp"

# Reliability knobs (not anti-detection bypass).
NAV_RETRIES = int(os.getenv("NAV_RETRIES", "3"))
STEP_SLEEP_MIN = float(os.getenv("STEP_SLEEP_MIN", "0.25"))
STEP_SLEEP_MAX = float(os.getenv("STEP_SLEEP_MAX", "0.75"))
TOTAL_TIMEOUT_S = int(os.getenv("TOTAL_TIMEOUT_S", "900"))


def wait_port(port: int, timeout_s: int = 90) -> str | None:
    start = time.time()
    while time.time() - start < timeout_s:
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


def jitter_sleep() -> None:
    time.sleep(random.uniform(STEP_SLEEP_MIN, STEP_SLEEP_MAX))


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    return (path.split("/", 1)[0] or "page").replace("-", "_")


def call_tool_retry(client: McpSseClient, name: str, arguments: dict | None = None) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, NAV_RETRIES + 1):
        try:
            jitter_sleep()
            return client.call_tool(name, arguments)
        except Exception as exc:
            last_exc = exc
            if attempt >= NAV_RETRIES:
                break
            time.sleep(min(2 * attempt, 6))
    raise RuntimeError(f"Tool call failed after {NAV_RETRIES} tries: {name}: {last_exc}")


def main() -> None:
    started = time.monotonic()
    today = date.today().isoformat()
    out_dir = MCP_ROOT / "out" / f"bbc_reliable_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "playwright-mcp.log"

    proc = subprocess.Popen(
        [
            "npx",
            "@playwright/mcp@latest",
            "--port",
            str(PLAYWRIGHT_PORT),
            "--allowed-hosts",
            # NOTE: this flag controls which *clients* may access the MCP server (Host header),
            # not which websites Playwright can navigate to. Restricting it to bbc.co.uk will
            # block local access to the server (403 on /sse). We keep it open here and enforce
            # target URLs in the script logic instead.
            "*",
            "--output-dir",
            str(out_dir),
            "--output-mode",
            "file",
            "--no-sandbox",
        ],
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    captured: list[dict] = []
    error: str | None = None
    try:
        host = wait_port(PLAYWRIGHT_PORT, timeout_s=90)
        if not host:
            raise RuntimeError("Playwright MCP server did not start")

        base_host = f"[{host}]" if ":" in host else host
        client = McpSseClient(f"http://{base_host}:{PLAYWRIGHT_PORT}")
        client.initialize()

        call_tool_retry(client, "browser_install")
        call_tool_retry(client, "browser_navigate", {"url": BASE_URL})
        call_tool_retry(client, "browser_wait_for", {"time": 2})

        # Basic "is this likely blocked?" check. We do not attempt to bypass CAPTCHAs.
        blocked_js = (
            "async (page) => {"
            "  const t = document.body ? (document.body.innerText || '') : '';"
            "  const s = t.toLowerCase();"
            "  return s.includes('captcha') || s.includes('verify you are human') || s.includes('access denied');"
            "}"
        )
        blocked = call_tool_retry(client, "browser_run_code", {"code": blocked_js})
        if blocked is True:
            raise RuntimeError("Blocked by an interstitial/CAPTCHA; stopping (no bypass attempted).")

        pages = [
            "https://www.bbc.co.uk/news",
            "https://www.bbc.co.uk/sport",
            "https://www.bbc.co.uk/weather",
        ]

        for url in pages:
            if time.monotonic() - started > TOTAL_TIMEOUT_S:
                raise RuntimeError("Total timeout reached")

            slug = slugify(url)
            md_name = f"mcp/out/bbc_reliable_{today}/bbc_{slug}.md"
            png_name = f"mcp/out/bbc_reliable_{today}/bbc_{slug}.png"

            call_tool_retry(client, "browser_navigate", {"url": url})
            call_tool_retry(client, "browser_wait_for", {"time": 2})

            blocked = call_tool_retry(client, "browser_run_code", {"code": blocked_js})
            if blocked is True:
                captured.append({"url": url, "blocked": True})
                continue

            call_tool_retry(client, "browser_snapshot", {"filename": md_name})
            call_tool_retry(client, "browser_take_screenshot", {"filename": png_name})

            captured.append(
                {
                    "url": url,
                    "blocked": False,
                    "markdown": md_name,
                    "screenshot": png_name,
                }
            )

        summary = {
            "date": today,
            "base_url": BASE_URL,
            "output_dir": str(out_dir),
            "pages": captured,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        error = str(exc)
        summary = {
            "date": today,
            "base_url": BASE_URL,
            "output_dir": str(out_dir),
            "pages": captured,
            "error": error,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
