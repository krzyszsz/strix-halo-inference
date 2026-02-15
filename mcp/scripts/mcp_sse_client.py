#!/usr/bin/env python
import json
import requests


class McpSseClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._id = 1
        # Some MCP servers (e.g. @playwright/mcp) implement a "legacy" SSE transport.
        # In practice, the stream may begin with `event: endpoint` and a `data:` line.
        # Be tolerant in parsing and keep the client simple/lightweight.
        self._sse = requests.get(self.base_url + "/sse", stream=True)
        self._lines = self._sse.iter_lines(decode_unicode=True)
        self._post_url = self.base_url + self._read_endpoint()

    @staticmethod
    def _line_to_str(line: str | bytes) -> str:
        if isinstance(line, bytes):
            return line.decode("utf-8", errors="replace")
        return line

    def _read_endpoint(self) -> str:
        for line in self._lines:
            text = self._line_to_str(line)
            # Accept both `data: ...` and `data:...` (some servers omit the space).
            if text.startswith("data:"):
                return text.split("data:", 1)[1].strip()
        raise RuntimeError("SSE endpoint not found")

    def _read_message(self) -> dict:
        for line in self._lines:
            text = self._line_to_str(line)
            if text.startswith("data:"):
                payload = text.split("data:", 1)[1].lstrip()
                if not payload:
                    continue
                return json.loads(payload)
        raise RuntimeError("SSE stream closed")

    def initialize(self) -> dict:
        msg = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "local-mcp-client", "version": "0.1.0"},
            },
        }
        self._id += 1
        requests.post(self._post_url, json=msg)
        result = self._read_message()
        # notify initialized
        requests.post(self._post_url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def call(self, method: str, params: dict | None = None) -> dict:
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        self._id += 1
        if params is not None:
            msg["params"] = params
        requests.post(self._post_url, json=msg)
        response = self._read_message()
        if "error" in response:
            raise RuntimeError(f"MCP error for {method}: {response['error']}")
        return response.get("result", {})

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self.call("tools/call", params=params)

    def list_tools(self) -> dict:
        return self.call("tools/list")

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        requests.post(self._post_url, json=msg)

    def close(self) -> None:
        try:
            self._sse.close()
        except Exception:
            pass
