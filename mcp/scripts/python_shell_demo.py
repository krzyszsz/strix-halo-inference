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
PY_PORT = int(os.getenv("PY_MCP_PORT", "8018"))
SH_PORT = int(os.getenv("SH_MCP_PORT", "8019"))
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "mcp"
HF_ROOT = os.getenv("HF_ROOT", "/mnt/hf")
MODEL_ROOT = os.getenv("MODEL_ROOT", os.path.join(HF_ROOT, "models"))


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


def llm_choose_tool(question: str) -> dict:
    prompt = (
        "Choose one tool and arguments to answer the question. "
        "Return ONLY JSON in the format {\"tool\": \"python_eval\"|\"run_shell\", \"args\": {...}}.\n\n"
        f"Question: {question}"
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
        resp = requests.post(f"{LLM_BASE}/chat/completions", json=payload, timeout=20)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return {}
    return {}


def main() -> Path:
    today = date.today().isoformat()
    out_dir = MCP_ROOT / "out" / f"python_shell_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_py = os.environ.copy()
    env_py.update({"MCP_TRANSPORT": "sse", "MCP_PORT": str(PY_PORT)})
    env_sh = os.environ.copy()
    env_sh.update({"MCP_TRANSPORT": "sse", "MCP_PORT": str(SH_PORT)})

    proc_py = subprocess.Popen(
        ["python", str(MCP_ROOT / "servers" / "local_python_mcp.py")],
        env=env_py,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc_sh = subprocess.Popen(
        ["python", str(MCP_ROOT / "servers" / "local_shell_mcp.py")],
        env=env_sh,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_port("127.0.0.1", PY_PORT, timeout=30):
            raise RuntimeError("Python MCP server did not start")
        if not wait_port("127.0.0.1", SH_PORT, timeout=30):
            raise RuntimeError("Shell MCP server did not start")

        client_py = McpSseClient(f"http://127.0.0.1:{PY_PORT}")
        client_sh = McpSseClient(f"http://127.0.0.1:{SH_PORT}")
        client_py.initialize()
        client_sh.initialize()

        questions = [
            "Compute the sum of squares from 1 to 20.",
            f"Show the size of {MODEL_ROOT} (human-readable).",
        ]

        results = []

        for q in questions:
            choice = llm_choose_tool(q)
            tool = choice.get("tool")
            args = choice.get("args", {})

            if tool == "python_eval":
                if "code" not in args:
                    args = {"code": "total = sum(i*i for i in range(1, 21))\nprint(total)"}
                result = client_py.call_tool("python_eval", args)
                if not result.get("structuredContent", {}).get("stdout"):
                    result = client_py.call_tool(
                        "python_eval",
                        {"code": "total = sum(i*i for i in range(1, 21))\nprint(total)"},
                    )
            elif tool == "run_shell":
                if "cmd" not in args:
                    args = {"cmd": f"du -sh '{MODEL_ROOT}'"}
                result = client_sh.call_tool("run_shell", args)
            else:
                if "size" in q:
                    result = client_sh.call_tool("run_shell", {"cmd": f"du -sh '{MODEL_ROOT}'"})
                    tool = "run_shell"
                else:
                    result = client_py.call_tool(
                        "python_eval",
                        {"code": "total = sum(i*i for i in range(1, 21))\nprint(total)"},
                    )
                    tool = "python_eval"

            results.append({
                "question": q,
                "tool": tool,
                "result": result,
            })

        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
        return out_dir
    finally:
        try:
            proc_py.terminate()
            proc_py.wait(timeout=5)
        except Exception:
            proc_py.kill()
        try:
            proc_sh.terminate()
            proc_sh.wait(timeout=5)
        except Exception:
            proc_sh.kill()


if __name__ == "__main__":
    main()
