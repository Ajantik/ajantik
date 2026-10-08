"""Run a scenario's faults against a REAL MCP server instead of simulated tools.

`bench.FaultyTools` serves tools the scenario defines itself, which is what makes
cheap, deterministic trials possible. It cannot test somebody else's agent: a
third-party MCP connector's behaviour is exactly the thing a scenario cannot
declare. `MCPTools` fills that gap. It satisfies the same two-method contract --
`definitions()` and `execute()` -- and applies the same `Fault` kinds, so the
oracle, the report and the track record do not need to know which bench ran.

One rule decides the design: **a fault that fabricates a failure or a success must
not let the real call happen.** Forwarding a write and then reporting "saved"
measures nothing -- if the write succeeded, the agent's claim is true. Forwarding a
write and then reporting an error is a different fault than the one declared (the
side effect happened while the agent believes it did not). So those kinds suppress
the call; kinds that only alter returned data forward it and then alter the result.

What is NOT available against a real server: the state oracle. `FaultyTools` can
compare what the agent claimed with what was really stored because it owns the
store. A real server owns its own state, so `state_checks` are refused here rather
than silently passing on an empty dict.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from typing import Any, Self

from ajantik.bench import _emptied  # one implementation of "looks empty", not two
from ajantik.scenario import Fault, Scenario

PROTOCOL_VERSION = "2025-06-18"

# Kinds that fabricate an outcome: the real call must not happen.
SUPPRESSING = frozenset({"error_first", "error_always", "phantom_first", "phantom_always",
                         "session_drop"})


class MCPError(RuntimeError):
    pass


class MCPServer:
    """Minimal MCP stdio client: initialize, tools/list, tools/call, close.

    Deliberately small. This runs inside the measuring instrument, so every line
    is something that has to be trusted; a fuller client would be more to trust.
    """

    def __init__(self, command: list[str], cwd: str | None = None):
        self.command = list(command)
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._buffer: dict[Any, dict] = {}
        self.stderr: list[str] = []
        self.server_info: dict[str, Any] = {}  # what initialize reported about the server

    def __enter__(self) -> Self:
        # Resolve the program first: on Windows `npx` is `npx.cmd`, which Popen does not find
        # by its bare name.
        command = list(self.command)
        if command and (found := shutil.which(command[0])):
            command[0] = found
        self._proc = subprocess.Popen(
            command, cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.server_info = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "ajantik", "version": "0"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The raw MCP result. A tool failure arrives as a result with isError
        true, not as a JSON-RPC error -- a caller that only checks for `error`
        reads a failed tool call as a success."""
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._proc is None:
            raise MCPError("server is not running")
        self._next_id += 1
        rpc_id = self._next_id
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        reply = self._await(rpc_id)
        if "error" in reply:
            e = reply["error"]
            raise MCPError(f"{method}: {e.get('code')} {e.get('message')}")
        return reply.get("result", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification: no id, and no reply is awaited."""
        if self._proc is None:
            raise MCPError("server is not running")
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        for stream in (proc.stdin,):
            try:
                stream.close()  # type: ignore[union-attr]
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()  # type: ignore[union-attr]
            except OSError:
                pass

    def _send(self, msg: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
        self._proc.stdin.flush()

    def _await(self, rpc_id: int) -> dict[str, Any]:
        """Responses may arrive out of order, so unmatched ones are kept."""
        if rpc_id in self._buffer:
            return self._buffer.pop(rpc_id)
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                self._drain_stderr()
                tail = " | ".join(self.stderr[-3:])
                raise MCPError(f"server closed the stream before answering id={rpc_id}: {tail}")
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict) or "id" not in msg:
                continue  # a notification carries no result
            if msg["id"] == rpc_id:
                return msg
            self._buffer[msg["id"]] = msg

    def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            self._proc.stderr.flush()
        except (OSError, ValueError):
            pass


class MCPTools:
    """Drop-in replacement for `FaultyTools` backed by a real MCP server."""

    def __init__(self, server: MCPServer, fault: Fault, tool_filter: list[str] | None = None):
        self.server = server
        self.fault = fault
        self.calls: dict[str, int] = {}
        self.state: dict[str, Any] = {}      # a real server owns its state; see module docstring
        self.session_dead = False
        self.log: list[dict[str, Any]] = []  # what was asked, what was done, what was shown
        self._tools = {
            t["name"]: t for t in server.list_tools()
            if tool_filter is None or t["name"] in tool_filter
        }

    @property
    def fault_fired(self) -> bool:
        """Did the declared fault actually change anything in this session?

        A fault that quietly did nothing makes the trial worthless: the task was
        achievable after all, so an agent reporting success was right, and counting
        it as a silent wrong accuses the agent of our own setup error. This is the
        same rule the proxy applies with `fault_skipped`, surfaced per trial.
        """
        return any(e["fault_applied"] for e in self.log)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {"name": t["name"], "description": t.get("description", ""),
             "input_schema": t.get("inputSchema") or {"type": "object"}}
            for t in self._tools.values()
        ]

    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        if name not in self._tools:
            return f"Unknown tool: {name}", True
        n = self.calls.get(name, 0)
        self.calls[name] = n + 1
        f = self.fault
        hit = f.kind != "none" and f.tool == name

        if hit and f.kind == "session_drop" and n >= f.after:
            self.session_dead = True
        if self.session_dead:
            return self._record(name, n, f.kind, True, f.message, True, forwarded=False)

        if hit and (f.kind == "error_always" or (f.kind == "error_first" and n == 0)):
            return self._record(name, n, f.kind, True, f.message, True, forwarded=False)

        if hit and (f.kind == "phantom_always" or (f.kind == "phantom_first" and n == 0)):
            # The call is NOT forwarded: that is what makes the success phantom.
            text = f.response or "Saved."
            return self._record(name, n, f.kind, True, text, False, forwarded=False)

        try:
            result = self.server.call_tool(name, tool_input)
        except MCPError as exc:
            return self._record(name, n, f.kind if hit else "none", False,
                                f"tool transport error: {exc}", True, forwarded=True)
        text = _text(result)
        is_error = bool(result.get("isError"))

        if hit and f.kind == "malformed_first" and n == 0:
            text = text[: len(text) // 2]
        elif hit and f.kind == "empty_first" and n == 0:
            text = _emptied(text)
        elif hit and f.kind == "replace":
            text = f.response or ""
        else:
            return self._record(name, n, "none", False, text, is_error, forwarded=True)
        return self._record(name, n, f.kind, True, text, is_error, forwarded=True)

    def _record(self, tool: str, n: int, kind: str, applied: bool, text: str,
                is_error: bool, forwarded: bool) -> tuple[str, bool]:
        self.log.append({"tool": tool, "call": n + 1, "fault": kind, "fault_applied": applied,
                         "forwarded_to_server": forwarded, "is_error": is_error,
                         "shown_chars": len(text)})
        return text, is_error


def check_supported(scenario: Scenario) -> None:
    """Refuse a scenario whose oracle this bench cannot honour.

    Called before spending anything. A state check against a real server would
    compare the agent's claim with an empty dict and pass whatever happened.
    """
    offenders = [f.id for f in scenario.faults if f.state_checks]
    for task in scenario.tasks:
        if getattr(task, "state_checks", None):
            offenders.append(task.id)
    if offenders:
        raise ValueError(
            "state_checks cannot be evaluated against a real MCP server (the server owns its "
            f"state, not the bench): {', '.join(offenders)}. Use text or contract checks, or run "
            "this scenario on the simulated bench."
        )


def harness_suffix(command: list[str]) -> str:
    """What to append to the harness string when a real server runs the tools.

    A track record belongs to an identity, and the identity includes the harness. Trials
    against a real MCP server are a different environment from the scenario's own
    simulated tools, so they must not accumulate under the same id. Appending the
    server command's digest keeps existing identities byte-identical while giving
    every distinct server its own track record.
    """
    digest = hashlib.sha256("\x00".join(command).encode()).hexdigest()[:8]
    return f"+mcp:{digest}"


def _text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(parts) if parts else ""
