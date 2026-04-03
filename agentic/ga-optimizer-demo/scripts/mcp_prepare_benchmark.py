#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


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


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    mcp_root = repo_root / "mcp"
    out_dir = repo_root / "agentic" / "ga-optimizer-demo" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "mcp_benchmark.json"

    sys.path.insert(0, str(mcp_root / "scripts"))
    from mcp_sse_client import McpSseClient  # noqa: E402

    port = int(os.getenv("GA_MCP_PORT", "8028"))
    env = os.environ.copy()
    env.update({"MCP_TRANSPORT": "sse", "MCP_PORT": str(port)})

    proc = subprocess.Popen(
        ["python", str(mcp_root / "servers" / "local_python_mcp.py")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_port("127.0.0.1", port, timeout=30):
            raise RuntimeError("Python MCP server did not start")

        client = McpSseClient(f"http://127.0.0.1:{port}")
        client.initialize()

        sphere_opt = client.call_tool(
            "python_eval",
            {"code": "x=[0.0,0.0,0.0]\nprint(sum(v*v for v in x))"},
        )
        sphere_probe = client.call_tool(
            "python_eval",
            {"code": "x=[1.0,-2.0,0.5]\nprint(sum(v*v for v in x))"},
        )

        def stdout_value(payload: dict) -> float:
            txt = payload.get("structuredContent", {}).get("stdout", "0").strip()
            return float(txt or "0")

        data = {
            "objective": "sphere",
            "dimensions": 3,
            "known_optimum_vector": [0.0, 0.0, 0.0],
            "known_optimum_score": stdout_value(sphere_opt),
            "probe_vector": [1.0, -2.0, 0.5],
            "probe_score": stdout_value(sphere_probe),
            "source": "local_python_mcp",
        }
        out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps(data, indent=2))
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
