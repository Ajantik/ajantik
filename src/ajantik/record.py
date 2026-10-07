"""Record one real run and turn it into a scenario, so nobody writes YAML by hand.

`scenario-from-server` reads a server's tool schemas, but the rest of a scenario -- what the
tools answer, which of them write, which read the writes back, what the world should
look like when the task is done -- still had to be written. All of that is visible in
one clean run of the agent against its real server, so it is recorded instead:

    agent  <-- MCP -->  recording proxy  <-- MCP -->  your real server
                              |
                          call log  -->  scenario.yaml

What comes from where, kept apart on purpose and written into the file:

  recorded  tool names, descriptions, input schemas, real replies, the final value of
            every write -- from the run itself
  inferred  which tools write (name and description verbs, as in `from_server`), which field of
            a write is the key and which the value, and which read tools read the writes
            back (their replies contained a value written earlier in the run)
  yours     whether that reference run was actually correct: the state checks are what
            the agent did, and an agent that did the wrong thing produces the wrong checks

The reference run calls the real server, with real side effects. Point it at a test
account or a sandbox.

    python -m ajantik.record --log /tmp/calls.jsonl -- <real MCP server command...>

runs the proxy itself; `ajantik record` (cli) runs the agent through it and writes the
scenario.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

from ajantik.from_server import infer_effect
from ajantik.mcp import PROTOCOL_VERSION, MCPServer

# The last names in each list are Turkish ("alan" field, "anahtar" key, "deger" value),
# kept so a server with Turkish argument names is still inferred the same way.
KEY_NAMES = ("field", "key", "name", "id", "path", "slug", "title", "alan", "anahtar")
VALUE_NAMES = ("value", "content", "text", "body", "data", "contents", "deger")
WHOLE_STATE = "__whole_state__"  # a read-back tool with no key argument returns everything


# -- the proxy -----------------------------------------------------------------


class RecordingProxy:
    """An MCP server to the agent, an MCP client to the real server, a log in between."""

    def __init__(self, real: MCPServer, log: TextIO):
        self.real = real
        self.log = log

    def _write(self, **row: Any) -> None:
        self.log.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log.flush()

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, rpc_id = msg.get("method"), msg.get("id")
        if method is None or rpc_id is None:
            return None  # notifications, and responses we never asked for
        if method == "initialize":
            # The real server's instructions are not passed on. A recorded scenario has
            # none either, so recording with them would describe a different run.
            return _ok(rpc_id, {"protocolVersion": PROTOCOL_VERSION,
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "ajantik-record", "version": "0"}})
        if method == "ping":
            return _ok(rpc_id, {})
        if method == "tools/list":
            tools = self.real.list_tools()
            self._write(event="tools", tools=tools)
            return _ok(rpc_id, {"tools": tools})
        if method == "tools/call":
            params = msg.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            result = self.real.call_tool(name, args)
            text = "\n".join(c.get("text", "") for c in result.get("content") or []
                             if isinstance(c, dict) and c.get("type") == "text")
            self._write(event="call", tool=name, arguments=args, text=text,
                        is_error=bool(result.get("isError")))
            return _ok(rpc_id, result)
        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": f"method not supported: {method}"}}

    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            reply = self.handle(msg) if isinstance(msg, dict) else None
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def _ok(rpc_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def proxy_command(log: Path, server: list[str], python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "ajantik.record", "--log", str(log), "--", *server]


# -- from a call log to a scenario ------------------------------------------------


def _props(tool: dict[str, Any]) -> list[str]:
    return list(((tool.get("inputSchema") or {}).get("properties") or {}).keys())


def key_value_fields(tool: dict[str, Any], calls: list[dict[str, Any]]) -> tuple[str, str, str]:
    """(key_field, value_field, basis) for a writing tool."""
    props = _props(tool) or sorted({k for c in calls for k in c["arguments"]})
    key = next((p for n in KEY_NAMES for p in props if p.lower() == n), None)
    value = next((p for n in VALUE_NAMES for p in props if p.lower() == n and p != key), None)
    if key and value:
        return key, value, f"named fields: key '{key}', value '{value}'"
    if len(props) >= 2:
        k, v = key or props[0], value or next(p for p in props if p != (key or props[0]))
        return k, v, f"guessed from order: key '{k}', value '{v}' -- CHECK"
    only = props[0] if props else "value"
    return only, only, f"single field '{only}' is both key and value -- CHECK"


def build_scenario(log_rows: list[dict[str, Any]], task: str,
                   server: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Return (scenario data, notes). Notes are every inference, for a person to check."""
    listed = next((r["tools"] for r in log_rows if r["event"] == "tools"), None)
    calls = [r for r in log_rows if r["event"] == "call"]
    if listed is None:
        raise ValueError("The agent never listed the tools: it did not reach the server.")
    if not calls:
        raise ValueError("The agent called no tool in the reference run; nothing to record.")
    notes: list[str] = []
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for c in calls:
        by_tool.setdefault(c["tool"], []).append(c)

    effects = {t["name"]: infer_effect(t) for t in listed}
    writers: dict[str, tuple[str, str]] = {}
    state: dict[str, Any] = {}
    written_values: list[str] = []
    for t in listed:
        if effects[t["name"]][0] == "write":
            k, v, basis = key_value_fields(t, by_tool.get(t["name"], []))
            writers[t["name"]] = (k, v)
            notes.append(f"{t['name']}: writes ({effects[t['name']][1]}); {basis}")

    # Replay the run in order: the final value of each write is the reference state, and a
    # read whose reply contains something written before it is a read-back.
    readbacks: dict[str, str] = {}
    # A tool already called before the first write is a source, even if a later call of it
    # echoes a written value (an agent re-reading its input after writing).
    first_write = next((i for i, c in enumerate(calls) if c["tool"] in writers), len(calls))
    sources = {c["tool"] for c in calls[:first_write]}
    for c in calls:
        if c["is_error"]:
            continue
        if c["tool"] in writers:
            k, v = writers[c["tool"]]
            if k in c["arguments"]:
                state[str(c["arguments"][k])] = c["arguments"].get(v)
                written_values.append(json.dumps(c["arguments"].get(v), ensure_ascii=False)
                                      .strip('"'))
        elif c["tool"] not in sources and any(w and w in c["text"] for w in written_values):
            shared = [p for p in c["arguments"] if any(p == kv[0] for kv in writers.values())]
            readbacks.setdefault(c["tool"], shared[0] if shared else WHOLE_STATE)

    tools = []
    for t in listed:
        name = t["name"]
        spec: dict[str, Any] = {"name": name, "description": (t.get("description") or "").strip(),
                                "input_schema": t.get("inputSchema") or
                                {"type": "object", "properties": {}}}
        ok = [c for c in by_tool.get(name, []) if not c["is_error"]]
        if name in writers:
            spec.update(effect="write", key_field=writers[name][0], value_field=writers[name][1])
        elif name in readbacks:
            spec.update(effect="read", key_field=readbacks[name],
                        value_field=next(iter(writers.values()))[1] if writers else "value")
            notes.append(f"{name}: reads the writes back (its reply contained a written value); "
                         "simulated replies use a normalised shape, not the real one")
        else:
            spec["effect"] = "none"
        if ok and name not in readbacks:
            spec["response"] = ok[0]["text"]
            if len({c["text"] for c in ok}) > 1:
                notes.append(f"{name}: answered differently across calls; the first reply is "
                             "used for every call")
        if not by_tool.get(name):
            notes.append(f"{name}: never called in the reference run; no recorded reply and "
                         "no fault will be tested on it")
        tools.append(spec)

    if not state:
        notes.append("No successful write was recorded: the scenario has no state checks, "
                     "so the world cannot be judged. Record a run in which the task succeeds.")
    data = {
        "skill": ".",
        "recorded_from": {"server": server,
                          "at": datetime.now(UTC).isoformat(timespec="seconds"),
                          "calls": len(calls)},
        "tools": tools,
        "tasks": [{"id": "task-1", "prompt": task,
                   "state_checks": [{"key": k, "equals": v} for k, v in state.items()]}],
        "faults": [{"id": "clean"}],
        "auto_faults": ["all"],
    }
    return data, notes


def render(data: dict[str, Any], notes: list[str]) -> str:
    head = [
        "# Recorded by `ajantik record` from one real run. REVIEW BEFORE SPENDING.",
        "#",
        "# recorded  tools, schemas, real replies, the final value of every write",
        "# inferred  (below) which tools write, key/value fields, which reads read back",
        "# yours     the state checks are what the agent DID in the reference run. If that",
        "#           run was wrong, so are they.",
        "#",
        *[f"# - {n}" for n in notes],
        "",
    ]
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)
    return "\n".join(head) + body


def record(server: list[str], task: str, agent_template: list[str], out_dir: Path,
           timeout_s: int = 300, python: str | None = None) -> tuple[Path, str]:
    """Run the agent once through the proxy. Returns (call log, agent stdout)."""
    from ajantik.rounds import _substitute

    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "reference-calls.jsonl").resolve()
    sandbox = out_dir / "reference-sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir()
    wall = shlex.join(proxy_command(log, server, python))
    command = _substitute(agent_template, wall, task, str(sandbox.resolve()))
    done = subprocess.run(command, cwd=sandbox, capture_output=True, text=True,
                          timeout=timeout_s, check=False)
    (out_dir / "reference-agent-stdout.txt").write_text(done.stdout or "", encoding="utf-8")
    (out_dir / "reference-agent-stderr.txt").write_text(done.stderr or "", encoding="utf-8")
    if not log.exists():
        raise SystemExit("The agent never started the recording proxy: check that the agent "
                         "command passes {wall} as an MCP server.\n" + (done.stderr or "")[-800:])
    return log, done.stdout or ""


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="ajantik.record", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("server", nargs=argparse.REMAINDER, help="-- <real MCP server command>")
    args = p.parse_args(argv)
    server = args.server[1:] if args.server[:1] == ["--"] else args.server
    if not server:
        raise SystemExit("STOP: no server command after `--`.")
    with MCPServer(server) as real, args.log.open("a", encoding="utf-8") as log:
        RecordingProxy(real, log).serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
