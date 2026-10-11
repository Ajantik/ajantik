"""Write a `--check` in a few lines: read the system through its own connector.

A check answers, after every run, what is wrong in the real system: a duplicate contact, a
missing row, a file that is not there. The verdict rests on it. The easiest way to look is the
way the skill looks: through the same MCP connector, by its read tools. `connector()` starts that
connector exactly as Claude Code has it configured for the project, local or remote, and never
through the fault proxy, so a check always sees the real state.

    #!/usr/bin/env python3
    from ajantik.check import connector, report

    problems = []
    with connector("crm") as crm:
        for email in ["a@x.com", "b@y.com"]:
            found = crm.json("search_contacts", {"query": email})
            if len(found) != 1:
                problems.append(f"{email}: {len(found)} contacts")
    report(problems)

Run it as `ajantik test run ... --check "python3 check.py"`.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ajantik.mcp import MCPError, MCPServer, _text


class Connector:
    """A started connector: call its tools, get their text or their JSON."""

    def __init__(self, client: Any, name: str):
        self.client, self.name = client, name

    def tools(self) -> list[str]:
        return [t["name"] for t in self.client.list_tools()]

    def text(self, tool: str, arguments: dict[str, Any] | None = None) -> str:
        result = self.client.call_tool(tool, arguments or {})
        if result.get("isError"):
            raise MCPError(f"{self.name}.{tool}: {_text(result)[:300]}")
        if isinstance(result.get("structuredContent"), (dict, list)):
            return json.dumps(result["structuredContent"])
        return _text(result)

    def json(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        """The tool's answer parsed as JSON (structured content first, then the text)."""
        return json.loads(self.text(tool, arguments))


@contextmanager
def connector(name: str, cwd: str | os.PathLike[str] = ".") -> Iterator[Connector]:
    """The MCP server `name` from Claude Code's config for `cwd`, started directly."""
    from ajantik.autotest import discover_servers, run_env
    from ajantik.mcp_http import AuthRequired, HTTPServer, bridge_command, expand

    servers, _ = discover_servers(Path(cwd).resolve())
    spec = servers.get(name)
    if spec is None:
        raise SystemExit(f"No MCP server named {name!r} for {Path(cwd).resolve()}. "
                         f"Found: {', '.join(servers) or 'none'}.")
    if spec.get("url"):
        url = expand(spec["url"])
        client: Any = HTTPServer(url, {k: expand(v) for k, v in spec["headers"].items()})
        try:
            client.__enter__()
        except AuthRequired:
            if spec["headers"]:
                raise
            client = MCPServer(bridge_command(url, "sse-only" if spec["type"] == "sse"
                                              else "http-first"), env=run_env()).__enter__()
    else:
        client = MCPServer([spec["command"], *spec.get("args", [])],
                           env={**run_env(), **(spec.get("env") or {})}).__enter__()
    try:
        yield Connector(client, name)
    finally:
        client.close()


def report(problems: Iterable[str]) -> None:
    """Print the one line `--check` reads, and nothing else on stdout after it."""
    print(json.dumps({"problems": [str(p) for p in problems]}, ensure_ascii=False))
    sys.stdout.flush()
