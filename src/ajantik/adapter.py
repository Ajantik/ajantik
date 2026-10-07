"""The wall, for agents that do not speak MCP.

Most agents built in code (a hand-written tool loop, LangChain, the OpenAI or Anthropic
SDK directly) take tools as a list of schemas and run them through one executor
function. Such an agent can be tested without MCP by swapping in two things from here:
the tool list, in the format its SDK expects, and the executor.

    from ajantik.adapter import Wall

    with Wall.from_env() as wall:            # set by `ajantik round --in-process`
        tools = wall.tools("openai")         # or "anthropic" / "mcp"
        ...
        text, is_error = wall.call(name, arguments)

Same fault semantics and the same session record as the MCP wall (`ajantik.wall`), because
it is the same object underneath: a verdict computed from an in-process round means
exactly what one computed from an MCP round means.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

from ajantik.scenario import Scenario, load_scenario
from ajantik.wall import FaultyServer, Transcript, pick_fault

ENV_SCENARIO = "AJANTIK_SCENARIO"
ENV_FAULT = "AJANTIK_FAULT"
ENV_RECORD = "AJANTIK_RECORD"
FORMATS = ("anthropic", "openai", "mcp")


class Wall:
    """One scenario, one fault, one session, in the agent's own process."""

    def __init__(self, scenario: str | Path, fault: str = "clean",
                 record: str | Path | None = None):
        scen = load_scenario(Path(scenario))
        self._server = FaultyServer(scen, pick_fault(scen, fault),
                                    Transcript(Path(record)) if record else Transcript())
        self._closed = False

    @classmethod
    def from_env(cls) -> Wall:
        """Built from the variables the round runner sets for an in-process agent."""
        missing = [v for v in (ENV_SCENARIO, ENV_FAULT, ENV_RECORD) if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Not started by `ajantik round --in-process`: missing "
                               f"{', '.join(missing)}. Construct Wall(scenario, fault, record) "
                               "directly instead.")
        return cls(os.environ[ENV_SCENARIO], os.environ[ENV_FAULT], os.environ[ENV_RECORD])

    @property
    def scenario(self) -> Scenario:
        return self._server.scenario

    def tools(self, fmt: str = "anthropic") -> list[dict[str, Any]]:
        """The scenario's tools in the shape the agent's SDK expects."""
        if fmt not in FORMATS:
            raise ValueError(f"Unknown tool format {fmt!r}; use one of {', '.join(FORMATS)}")
        specs = self._server.scenario.tools
        if fmt == "anthropic":
            return [{"name": t.name, "description": t.description,
                     "input_schema": t.input_schema} for t in specs]
        if fmt == "openai":
            return [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in specs]
        return self._server.definitions()

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, bool]:
        """Run one tool call. Returns (text, is_error), as an MCP tool result would."""
        if self._closed:
            raise RuntimeError("This session is closed; a new trial needs a new Wall.")
        return self._server.call_tool(name, arguments or {})

    def close(self) -> dict[str, Any]:
        """Write the final state. Idempotent; the record is complete only after this."""
        if self._closed:
            return {}
        self._closed = True
        return self._server.finish()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False
