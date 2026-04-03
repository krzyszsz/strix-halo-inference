#!/usr/bin/env python
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict

from fastmcp import FastMCP

mcp = FastMCP("local-python")


@mcp.tool
def python_eval(code: str) -> Dict[str, Any]:
    """Execute Python code and return stdout and simple locals."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    locals_ns: Dict[str, Any] = {}
    globals_ns: Dict[str, Any] = {}
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exec(code, globals_ns, locals_ns)
    safe_locals = {}
    for k, v in locals_ns.items():
        if k.startswith("_"):
            continue
        try:
            safe_locals[k] = repr(v)
        except Exception:
            safe_locals[k] = "<unrepr-able>"
    return {
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "locals": safe_locals,
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8018"))
    if transport in {"sse", "streamable-http"}:
        mcp.run(transport=transport, host=host, port=port, show_banner=False)
    else:
        mcp.run(show_banner=False)
