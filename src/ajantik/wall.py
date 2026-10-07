"""Serve a scenario's faulty tools over MCP stdio, to an agent we do not own.

Euro NCAP does not modify the car. This process is the wall. A third-party MCP
client (Goose, Cline, any agent) connects over stdio, lists the tools the
scenario defines, and calls them. One fault is applied, by the same
`bench.FaultyTools` the lab's own harness uses, so the fault semantics have
one implementation and not two. We never see the agent's code, its prompt, or
its model key, and the agent never learns which fault is in effect.

Every call is appended to a JSONL transcript together with the tool state after
it. That is what makes a phantom write legible: the tool said "saved" and
the state stayed empty, recorded on our side, rather than the agent's word
against ours. The final state is written when the stream closes, and a state
check is evaluated against it.

stdout is the protocol channel and carries nothing else; notes go to stderr.

    python -m ajantik.wall --scenario examples/intake-form/scenario.yaml \
        --fault "phantom-success:set_field" --record /tmp/session.jsonl
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ajantik.bench import FaultyTools
from ajantik.mcp import PROTOCOL_VERSION
from ajantik.scenario import Fault, Scenario, load_scenario

SERVER_NAME = "ajantik-server"
SERVER_VERSION = "0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Transcript:
    """Append-only record of what the world told the agent."""

    def __init__(self, path: Path | None = None):
        self._fh: TextIO | None = path.open("w", encoding="utf-8") if path else None
        self.entries: list[dict[str, Any]] = []

    def add(self, **entry: Any) -> dict[str, Any]:
        row = {"seq": len(self.entries) + 1, "at": _now(), **entry}
        self.entries.append(row)
        if self._fh is not None:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()
        return row

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class FaultyServer:
    """The MCP server half. One scenario, one fault, one session."""

    def __init__(self, scenario: Scenario, fault: Fault, transcript: Transcript | None = None):
        self.scenario = scenario
        self.fault = fault
        self.transcript = transcript or Transcript()
        # skill is None: read_skill_file is a lab-harness affordance and is never
        # advertised or reachable here, so no skill is needed or exposed.
        self.tools = FaultyTools(None, scenario.tools, fault, scenario.initial_state)  # type: ignore[arg-type]
        self.names = {t.name for t in scenario.tools}
        self.initialized = False
        self.finished: dict[str, Any] | None = None
        # success_possible travels with the record: under some faults the task cannot
        # succeed and the right behaviour is to say so, not to succeed. A judge that
        # scores the end state without knowing this fails a correct agent.
        self.transcript.add(event="session_start", fault=fault.id, fault_kind=fault.kind,
                            fault_tool=fault.tool, success_possible=fault.success_possible,
                            tools=sorted(self.names),
                            initial_state=dict(scenario.initial_state))

    # -- protocol ----------------------------------------------------------

    def definitions(self) -> list[dict[str, Any]]:
        """MCP spells it inputSchema; the scenario carries the Anthropic spelling."""
        return [{"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in self.scenario.tools]

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """One request in, one response out. None for a notification."""
        method = msg.get("method")
        rpc_id = msg.get("id")
        params = msg.get("params") or {}

        if method is None:
            return self._error(rpc_id, INVALID_REQUEST, "no method")
        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                self.initialized = True
            return None
        if rpc_id is None:
            return None  # an unknown notification: nothing to answer

        if method == "initialize":
            # No `instructions`: the wall must not whisper to the car. Anything
            # we said here would steer the agent and contaminate the measurement.
            return self._ok(rpc_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "ping":
            return self._ok(rpc_id, {})
        if method == "tools/list":
            return self._ok(rpc_id, {"tools": self.definitions()})
        if method == "tools/call":
            return self._ok(rpc_id, self._call(params))
        return self._error(rpc_id, METHOD_NOT_FOUND, f"method not supported: {method}")

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        text, is_error = self.call_tool(params.get("name"), params.get("arguments") or {})
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    def call_tool(self, name: Any, arguments: dict[str, Any]) -> tuple[str, bool]:
        """The transport-free core: MCP, HTTP and the in-process adapter all call this,
        so a phantom save means the same thing whichever way the agent reaches the wall."""
        if name not in self.names:
            # Also keeps read_skill_file unreachable, so the None skill is never touched.
            text = f"Unknown tool: {name}"
            self.transcript.add(event="tool_call_rejected", tool=name, arguments=arguments,
                                text=text)
            return text, True
        calls_before = self.tools.calls.get(name, 0)
        text, is_error = self.tools.execute(name, arguments)
        self.transcript.add(event="tool_call", tool=name, arguments=arguments,
                            calls_before=calls_before, is_error=is_error, text=text,
                            state_after=dict(self.tools.state),
                            session_dead=self.tools.session_dead)
        return text, is_error

    @staticmethod
    def _ok(rpc_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

    @staticmethod
    def _error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}

    # -- loop --------------------------------------------------------------

    def serve(self, stdin: TextIO, stdout: TextIO) -> dict[str, Any]:
        """Read newline-delimited JSON until the stream closes. Returns the final state."""
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError as exc:
                self._emit(stdout, self._error(None, PARSE_ERROR, f"invalid JSON: {exc}"))
                continue
            if not isinstance(msg, dict):
                self._emit(stdout, self._error(None, INVALID_REQUEST, "not a JSON-RPC object"))
                continue
            try:
                reply = self.handle(msg)
            except Exception as exc:  # noqa: BLE001 - a crashed wall must not look like a passing run
                reply = self._error(msg.get("id"), INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
                self.transcript.add(event="server_error", error=f"{type(exc).__name__}: {exc}")
            if reply is not None:
                self._emit(stdout, reply)
        return self.finish()

    def finish(self, ended_by: str = "eof") -> dict[str, Any]:
        """Write the final state once. `ended_by` says how the session closed."""
        if self.finished is not None:
            return self.finished
        self.finished = self.transcript.add(
            event="session_end", final_state=dict(self.tools.state),
            session_dead=self.tools.session_dead, calls=dict(self.tools.calls), ended_by=ended_by)
        self.transcript.close()
        return self.finished

    @staticmethod
    def _emit(stdout: TextIO, msg: dict[str, Any]) -> None:
        stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        stdout.flush()


def pick_fault(scenario: Scenario, fault_id: str) -> Fault:
    from ajantik.faults import canonical_id

    fault_id = canonical_id(fault_id)
    for f in scenario.faults:
        if f.id == fault_id:
            return f
    known = ", ".join(f.id for f in scenario.faults)
    raise SystemExit(f"Unknown fault: {fault_id}. The scenario has: {known}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ajantik.wall", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, type=Path, dest="scenario",
                   help="path to the scenario YAML")
    p.add_argument("--fault", default="clean", dest="fault",
                   help="fault id to apply (default: clean)")
    p.add_argument("--record", type=Path, dest="record",
                   help="file the JSONL session record is written to")
    p.add_argument("--list-faults", action="store_true",
                   dest="list_faults", help="print the scenario's faults and exit")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario)
    if args.list_faults:
        for f in scenario.faults:
            print(f"{f.id}\t{f.kind}\t{f.tool or '-'}\t{f.description}")
        return 0

    server = FaultyServer(scenario, pick_fault(scenario, args.fault),
                          Transcript(args.record) if args.record else Transcript())

    # Some clients (Claude Code among them) stop their MCP servers with a signal rather
    # than by closing stdin. The final state must be written either way: a session with no
    # session_end cannot be judged, and every trial of such an agent would be lost.
    def stop(signum: int, _frame: Any) -> None:
        server.finish(ended_by=signal.Signals(signum).name)
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    print(f"{SERVER_NAME}: {len(scenario.tools)} tools, fault={args.fault}", file=sys.stderr)
    server.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
