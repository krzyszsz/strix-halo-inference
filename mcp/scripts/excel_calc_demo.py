#!/usr/bin/env python
import json
import os
import re
import socket
import subprocess
import time
from datetime import date
from pathlib import Path

import requests

from mcp_sse_client import McpSseClient

LLM_BASE = os.getenv("LLM_BASE", "http://127.0.0.1:8004/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-gguf")
MCP_PORT = int(os.getenv("EXCEL_MCP_PORT", "8017"))
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "mcp"


def wait_port(host: str, port: int, timeout: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket() as s:
            s.settimeout(1)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(1)
    return False


def llm_pick_values() -> list[int]:
    prompt = (
        "Return a JSON array of 5 small integers (1-20). "
        "Only output the JSON array."
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise JSON generator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 64,
    }
    try:
        resp = requests.post(f"{LLM_BASE}/chat/completions", json=payload, timeout=20)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            vals = json.loads(match.group(0))
            if isinstance(vals, list) and all(isinstance(v, int) for v in vals):
                return vals[:5]
    except Exception:
        pass
    return [3, 5, 8, 13, 21]


def parse_tool_number(result: dict) -> float:
    if not isinstance(result, dict):
        return float(result)
    if result.get("structuredContent") is not None:
        try:
            return float(result["structuredContent"])
        except Exception:
            pass
    content = result.get("content", [])
    if content and isinstance(content, list):
        text = content[0].get("text", "")
        try:
            return float(text)
        except Exception:
            pass
    return 0.0


def main() -> Path:
    today = date.today().isoformat()
    out_dir = MCP_ROOT / "out" / f"excel_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wb_path = out_dir / "demo.xlsx"

    env = os.environ.copy()
    env.update({"MCP_TRANSPORT": "sse", "MCP_PORT": str(MCP_PORT)})
    proc = subprocess.Popen(
        ["python", str(MCP_ROOT / "servers" / "local_excel_mcp.py")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_port("127.0.0.1", MCP_PORT, timeout=30):
            raise RuntimeError("Excel MCP server did not start")

        client = McpSseClient(f"http://127.0.0.1:{MCP_PORT}")
        client.initialize()

        values = llm_pick_values()

        client.call_tool("excel_new", {"path": str(wb_path)})
        for idx, val in enumerate(values, start=1):
            cell = f"A{idx}"
            client.call_tool(
                "excel_set_cell",
                {"path": str(wb_path), "sheet": "Sheet1", "cell": cell, "value": val},
            )

        sum_result = client.call_tool(
            "excel_sum_range",
            {"path": str(wb_path), "sheet": "Sheet1", "cell_range": f"A1:A{len(values)}"},
        )
        total = parse_tool_number(sum_result)

        client.call_tool(
            "excel_set_cell",
            {"path": str(wb_path), "sheet": "Sheet1", "cell": "B1", "value": "Sum"},
        )
        client.call_tool(
            "excel_set_cell",
            {"path": str(wb_path), "sheet": "Sheet1", "cell": "B2", "value": total},
        )

        summary = {
            "values": values,
            "sum": total,
            "workbook": str(wb_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return wb_path
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
