"""The scenario's own simulated tools, and the one fault applied to them.

Split out of `trial` on purpose: this is the world the agent acts in, and it
must run without a model SDK. `wall` serves exactly this over MCP to an agent
we do not own, and `trial` calls it directly for the lab's own harness. One
implementation of the fault semantics, not two.
"""

from __future__ import annotations

import json
from typing import Any

from ajantik.scenario import Fault, ToolSpec
from ajantik.skill import Skill

READ_SKILL_FILE = "read_skill_file"

def _emptied(text: str) -> str:
    """Same shape, nothing in it: what a result looks like before it has loaded."""
    try:
        data = json.loads(text)
    except ValueError:
        # Agent-visible placeholder for a result that is not JSON. Recorded runs keep the
        # reply they actually got, so changing it does not alter any recorded verdict.
        return '{"status": "loading"}'

    def empty(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: empty(x) for k, x in v.items()}
        if isinstance(v, list):
            return []
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return 0
        if isinstance(v, str):
            return ""
        return v

    return json.dumps(empty(data), ensure_ascii=False)

class FaultyTools:
    """Serves the scenario's simulated tools, keeps their state, and applies one fault."""

    def __init__(
        self, skill: Skill, tools: list[ToolSpec], fault: Fault, initial_state: dict | None = None
    ):
        self.skill = skill
        self.tools = {t.name: t for t in tools}
        self.fault = fault
        self.calls: dict[str, int] = {}
        self.state: dict[str, Any] = dict(initial_state or {})
        self.session_dead = False

    def definitions(self) -> list[dict[str, Any]]:
        defs = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self.tools.values()
        ]
        if self.skill.extra_files():
            defs.append(
                {
                    "name": READ_SKILL_FILE,
                    "description": "Read a file bundled with the installed skill.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "enum": self.skill.extra_files()}},
                        "required": ["path"],
                    },
                }
            )
        return defs

    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        if name == READ_SKILL_FILE:
            try:
                return self.skill.read(str(tool_input.get("path", ""))), False
            except KeyError:
                return "File not found in skill.", True
        if name not in self.tools:
            return f"Unknown tool: {name}", True
        n = self.calls.get(name, 0)
        self.calls[name] = n + 1
        spec, f = self.tools[name], self.fault
        hit = f.kind != "none" and f.tool == name
        if hit and f.kind == "session_drop" and n >= f.after:
            self.session_dead = True
        if self.session_dead:
            return f.message, True
        if hit and (f.kind == "error_always" or (f.kind == "error_first" and n == 0)):
            return f.message, True

        if spec.effect == "write":
            phantom = hit and (f.kind == "phantom_always" or (f.kind == "phantom_first" and n == 0))
            if not phantom:
                self.state[str(tool_input.get(spec.key_field))] = tool_input.get(spec.value_field)
            text = spec.response
        elif spec.effect == "read":
            key = tool_input.get(spec.key_field)
            if key is None:
                text = json.dumps(self.state, ensure_ascii=False)
            else:
                text = json.dumps(
                    {spec.key_field: key, spec.value_field: self.state.get(str(key))}, ensure_ascii=False
                )
        else:
            text = spec.response

        if hit and n == 0 and f.kind == "malformed_first":
            text = text[: len(text) // 2]
        elif hit and n == 0 and f.kind == "empty_first":
            text = _emptied(text)
        elif hit and f.kind == "replace":
            text = f.response or ""
        return text, False
