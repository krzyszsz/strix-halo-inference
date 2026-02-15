#!/usr/bin/env python
import os
import subprocess
from typing import Any, Dict

from fastmcp import FastMCP

mcp = FastMCP("local-shell")


@mcp.tool
def run_shell(cmd: str, timeout: int = 30) -> Dict[str, Any]:
    """Run a shell command with a timeout and return output."""
    completed = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8019"))
    if transport in {"sse", "streamable-http"}:
        mcp.run(transport=transport, host=host, port=port, show_banner=False)
    else:
        mcp.run(show_banner=False)
