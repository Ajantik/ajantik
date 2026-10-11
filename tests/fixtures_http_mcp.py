"""A tiny remote MCP server (Streamable HTTP) used only by the tests.

The same store as `fixtures_mcp_server.py`, behind a URL: a bearer token, a session id handed
out on initialize and required afterwards, and answers as JSON or as server-sent events.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

TOOLS = [
    {"name": "store_value", "description": "store a value",
     "inputSchema": {"type": "object", "properties": {"field": {"type": "string"},
                                                      "value": {"type": "string"}}}},
    {"name": "read_all", "description": "read the whole store",
     "inputSchema": {"type": "object", "properties": {}}},
]


class RemoteStore:
    def __init__(self, token: str | None = "s3cret", sse: bool = False):
        self.token, self.sse = token, sse
        self.store: dict[str, Any] = {}
        self.sessions: set[str] = set()
        self.seen_headers: list[dict[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_DELETE(self) -> None:
                outer.sessions.discard(self.headers.get("Mcp-Session-Id", ""))
                self.send_response(204)
                self.end_headers()

            def do_POST(self) -> None:
                outer.seen_headers.append(dict(self.headers))
                if outer.token and self.headers.get("Authorization") != f"Bearer {outer.token}":
                    self.send_response(401)
                    self.end_headers()
                    return
                msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                method, rpc_id = msg.get("method"), msg.get("id")
                sid = self.headers.get("Mcp-Session-Id")
                if method != "initialize" and sid not in outer.sessions:
                    self.send_response(400)
                    self.end_headers()
                    return
                if rpc_id is None:
                    self.send_response(202)
                    self.end_headers()
                    return
                headers = {}
                if method == "initialize":
                    sid = f"session-{len(outer.sessions) + 1}"
                    outer.sessions.add(sid)
                    headers["Mcp-Session-Id"] = sid
                    result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                              "serverInfo": {"name": "remote-store", "version": "0"}}
                elif method == "tools/list":
                    result = {"tools": TOOLS}
                elif method == "tools/call":
                    p = msg.get("params") or {}
                    if p.get("name") == "store_value":
                        a = p.get("arguments") or {}
                        outer.store[str(a.get("field"))] = a.get("value")
                        text = "Stored (remote)."
                    else:
                        text = json.dumps(outer.store)
                    result = {"content": [{"type": "text", "text": text}]}
                else:
                    result = {}
                body = {"jsonrpc": "2.0", "id": rpc_id, "result": result}
                self.send_response(200)
                for k, v in headers.items():
                    self.send_header(k, v)
                if outer.sse:
                    note = {"jsonrpc": "2.0", "method": "notifications/progress",
                            "params": {"progress": 1}}
                    raw = (f"event: message\ndata: {json.dumps(note)}\n\n"
                           f"event: message\ndata: {json.dumps(body)}\n\n").encode()
                    self.send_header("Content-Type", "text/event-stream")
                else:
                    raw = json.dumps(body).encode()
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"

    def __enter__(self) -> Self:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
