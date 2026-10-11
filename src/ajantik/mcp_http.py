"""A remote MCP server over Streamable HTTP, behind the same small interface as `MCPServer`.

Most business systems offer their connector as a remote server (a URL), not a command. The
proxy needs to be the client of that server exactly as it is of a local one: `initialize`,
`tools/list`, `tools/call`, notifications. Streamable HTTP (MCP 2025-03-26 and later) is one
POST per message; the answer comes back either as JSON or as a short server-sent event stream.
The session id the server hands out on `initialize` goes on every later request.

Standard library only, for the same reason as `MCPServer`: this runs inside the measuring
instrument. Servers that need an OAuth login, or that only speak the older SSE transport, are
reached through `mcp-remote` instead (see `bridge_command`), which does that login itself.

Headers often carry a credential. They are never logged, never written to a lab, and never put
on a command line here: the proxy reads them from Claude Code's own configuration at start.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Self

from ajantik.mcp import PROTOCOL_VERSION, MCPError

BRIDGE = "mcp-remote@0.14.3"  # pinned: a bridge that changes under us changes the measurement


class AuthRequired(MCPError):
    """The server answered 401/403: it wants a login this client does not do."""


def expand(value: str, env: dict[str, str] | None = None) -> str:
    """Claude Code's `${VAR}` and `${VAR:-default}` in a URL or a header value."""
    env = os.environ if env is None else env

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(3)
        return env.get(name) or (default if default is not None else "")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}", sub, value)


def bridge_command(url: str, transport: str = "http-first") -> list[str]:
    """`mcp-remote` as a local stdio server in front of the URL: it does OAuth in the browser
    once and keeps the token in ~/.mcp-auth, so later headless runs need no login."""
    return ["npx", "-y", BRIDGE, url, "--transport", transport]


class HTTPServer:
    """Minimal Streamable HTTP MCP client with the interface of `MCPServer`."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 120):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.session: str | None = None
        self.protocol = PROTOCOL_VERSION
        self.server_info: dict[str, Any] = {}
        self.stderr: list[str] = []  # for parity with MCPServer
        self._next_id = 0
        self._open = False

    def __enter__(self) -> Self:
        self._open = True
        self.server_info = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "ajantik", "version": "0"},
        })
        self.protocol = str(self.server_info.get("protocolVersion") or PROTOCOL_VERSION)
        self.notify("notifications/initialized")
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:  # remote servers page their tool lists more often than local ones
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += result.get("tools", [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._open:
            raise MCPError("server is not connected")
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        reply = self._post(msg, self._next_id)
        if reply is None:
            raise MCPError(f"{method}: the server sent no answer")
        if "error" in reply:
            e = reply["error"]
            raise MCPError(f"{method}: {e.get('code')} {e.get('message')}")
        return reply.get("result", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self._open:
            raise MCPError("server is not connected")
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._post(msg, None)

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        if self.session:  # tell the server the session is over; it may not support it
            try:
                req = urllib.request.Request(self.url, method="DELETE",
                                             headers=self._headers())
                urllib.request.urlopen(req, timeout=10).close()
            except (urllib.error.URLError, OSError, ValueError):
                pass

    def _headers(self) -> dict[str, str]:
        h = {**self.headers, "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": self.protocol}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        return h

    def _post(self, msg: dict[str, Any], rpc_id: int | None) -> dict[str, Any] | None:
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode(), method="POST",
                                     headers=self._headers())
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AuthRequired(f"{msg.get('method')}: HTTP {exc.code}; the server wants "
                                   "a login") from exc
            body = exc.read(300).decode("utf-8", "replace")
            raise MCPError(f"{msg.get('method')}: HTTP {exc.code} {body}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise MCPError(f"{msg.get('method')}: {exc}") from exc
        with resp:
            if sid := resp.headers.get("Mcp-Session-Id"):
                self.session = sid
            if rpc_id is None or resp.status == 202:
                return None
            kind = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if kind == "text/event-stream":
                return _from_events(resp, rpc_id)
            raw = resp.read()
        if not raw.strip():
            return None
        doc = json.loads(raw)
        for m in doc if isinstance(doc, list) else [doc]:
            if isinstance(m, dict) and m.get("id") == rpc_id:
                return m
        return None


def _from_events(stream: Any, rpc_id: int) -> dict[str, Any] | None:
    """Read server-sent events until the one that answers `rpc_id`. Others (progress,
    log notifications) are skipped."""
    data: list[str] = []
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
            continue
        if line or not data:
            continue
        try:
            msg = json.loads("\n".join(data))
        except ValueError:
            msg = None
        data = []
        if (found := _answer(msg, rpc_id)) is not None:
            return found
    if data:  # a stream that ended without the closing blank line
        try:
            return _answer(json.loads("\n".join(data)), rpc_id)
        except ValueError:
            return None
    return None


def _answer(msg: Any, rpc_id: int) -> dict[str, Any] | None:
    for m in msg if isinstance(msg, list) else [msg]:
        if isinstance(m, dict) and m.get("id") == rpc_id and ("result" in m or "error" in m):
            return m
    return None
