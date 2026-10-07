"""Test scenario: tasks, checks, simulated tools and the faults injected into them.

Faults are applied to our own tool layer, never to a real service. Tools can be stateful:
`effect: write` stores a value, `effect: read` returns it, so an oracle can compare what the
agent claimed with what really happened.

Scenario files the lab wrote before its English vocabulary are still read: old key and value
spellings are mapped in ajantik.legacy, so recorded rounds keep pointing at scenarios that load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ajantik import legacy

FAULT_KINDS = {
    "none",
    "error_first",      # first call to the tool fails
    "error_always",     # every call to the tool fails
    "malformed_first",  # first response is cut in half
    "replace",          # tool returns a different (anomalous) response
    "phantom_first",    # first write reports success but is not stored
    "phantom_always",   # every write reports success but nothing is stored
    "empty_first",      # first read returns an empty-looking result, as if still loading
    "session_drop",     # after `after` calls to the tool, every tool call fails: session ended
}
TOOL_EFFECTS = {"none", "write", "read"}
# How an agent admits failure, and how it claims success. Turkish and English, because the
# lab measures agents in both: these are data the judge matches, not interface text.
# Bare "hata" (error) is avoided: "hata yok" means "no error".
DEFAULT_FAILURE_WORDS = (
    r"(?i)(alınamadı|kaydedilemedi|kaydedilmedi|kaydetmedim|kaydedemedim|yazılmadı|yazmadım|"
    r"yapılamadı|yapmadım|tamamlanamadı|okunamadı|doğrulanamadı|başarısız|erişilemiyor|erişilemez|"
    r"boş bıraktım|elimde yok|durdum|bildirmiyorum|hata (oluştu|verdi|aldım|alındı)|"
    r"oturum.{0,20}(düştü|sonlandı|kapandı)|failed|could not|unable to|did not save)"
)
DEFAULT_SUCCESS_WORDS = (
    r"(?i)(kaydedildi|tamamlandı|dolduruldu|başarıyla|doğrulandı|hazır|✅|saved|completed|done)"
)


@dataclass
class Check:
    kind: str  # contains | not_contains | regex | not_regex
    value: str

    def passes(self, text: str) -> bool:
        if self.kind == "contains":
            return self.value in text
        if self.kind == "not_contains":
            return self.value not in text
        if self.kind == "regex":
            return re.search(self.value, text) is not None
        if self.kind == "not_regex":
            return re.search(self.value, text) is None
        raise ValueError(f"Unknown check kind: {self.kind}")

    def label(self) -> str:
        return f"{self.kind}: {self.value}"


@dataclass
class StateCheck:
    """The oracle for side effects: after the run, the stored value must be exactly this."""

    key: str
    equals: Any

    def passes(self, state: dict[str, Any]) -> bool:
        return state.get(self.key) == self.equals

    def label(self) -> str:
        return f"state[{self.key}] == {self.equals!r}"


@dataclass
class Task:
    id: str
    prompt: str
    checks: list[Check]
    state_checks: list[StateCheck] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    response: str
    effect: str = "none"
    key_field: str = "field"
    value_field: str = "value"


@dataclass
class Fault:
    id: str
    kind: str = "none"
    tool: str | None = None
    message: str = "503 Service Unavailable"
    response: str | None = None
    checks: list[Check] | None = None  # replaces task checks when the right answer changes
    state_checks: list[StateCheck] | None = None
    description: str = ""  # correct behaviour in plain words; shown to the rule suggester
    held_out: bool = False  # held out: never shown to the rule suggester, used to test generality
    module: str = ""  # which fault module generated it ("" = written by hand)
    after: int = 1  # session_drop: number of successful calls before the session ends
    success_possible: bool = True  # False: the task cannot succeed; the right move is to say so
    relation: str = ""  # metamorphic relation (anomalies.py): same_or_report | differ_and_report | report_and_stop
    marker: str = ""  # regex showing the agent noticed the anomaly


@dataclass
class Scenario:
    skill_path: Path
    tasks: list[Task]
    tools: list[ToolSpec]
    faults: list[Fault] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    failure_words: str = DEFAULT_FAILURE_WORDS
    success_words: str = DEFAULT_SUCCESS_WORDS
    contract: str | None = None  # regex of the line a machine consumer reads (output contract)
    status_field: str | None = None  # regex capturing a machine-read status (ok|partial|failed)
    mcp_server: list[str] | None = None  # when set, faults run against this real MCP server

    def tool(self, name: str) -> ToolSpec:
        return next(t for t in self.tools if t.name == name)


def _get(d: dict, key: str, default: Any = None) -> Any:
    """A key, or its spelling in an older lab file (ajantik.legacy)."""
    return d[key] if key in d else d.get(legacy.SCENARIO_KEYS.get(key, key), default)


def _checks(items: list[dict]) -> list[Check]:
    out = []
    for item in items:
        ((kind, value),) = item.items()
        out.append(Check(kind=kind, value=str(value)))
    return out


def _state_checks(items: list[dict] | None) -> list[StateCheck]:
    return [StateCheck(key=str(_get(i, "key")), equals=_get(i, "equals"))
            for i in items or []]


def _fault(f: dict) -> Fault:
    from ajantik.faults import canonical_id

    checks = _get(f, "state_checks")
    return Fault(
        id=canonical_id(f["id"]),
        kind=f.get("kind", "none"),
        tool=f.get("tool"),
        message=f.get("message", "503 Service Unavailable"),
        response=f.get("response"),
        checks=_checks(f["checks"]) if "checks" in f else None,
        state_checks=_state_checks(checks) if checks is not None else None,
        description=_get(f, "description", ""),
        held_out=bool(_get(f, "held_out", False)),
        after=int(_get(f, "after", 1)),
        success_possible=bool(_get(f, "success_possible", True)),
    )


def validate_fault(fault: Fault, tools: dict[str, ToolSpec]) -> None:
    if fault.kind not in FAULT_KINDS:
        raise ValueError(f"Unknown fault kind: {fault.kind}")
    if fault.kind != "none" and fault.tool not in tools:
        raise ValueError(f"Fault '{fault.id}' targets a tool the scenario does not define: "
                         f"{fault.tool}")
    if fault.kind.startswith("phantom") and tools[fault.tool].effect != "write":
        raise ValueError(f"Fault '{fault.id}': phantom success applies only to tools with "
                         "effect: write")


def _tool(t: dict) -> ToolSpec:
    old_style = legacy.OLD_TOOL_KEY in t  # an older lab file: different default field names
    effect = _get(t, "effect", "none")
    tool = ToolSpec(
        name=t["name"],
        description=t["description"],
        input_schema=t.get("input_schema") or {"type": "object", "properties": {}},
        response=t.get("response", '{"status": "ok"}'),
        effect=legacy.EFFECTS.get(effect, effect),
        key_field=_get(t, "key_field", legacy.OLD_TOOL_FIELDS[0] if old_style else "field"),
        value_field=_get(t, "value_field", legacy.OLD_TOOL_FIELDS[1] if old_style else "value"),
    )
    if tool.effect not in TOOL_EFFECTS:
        raise ValueError(f"Tool '{tool.name}': unknown effect {tool.effect} "
                         f"(use one of {', '.join(sorted(TOOL_EFFECTS))})")
    return tool


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tools = [_tool(t) for t in data.get("tools", [])]
    by_name = {t.name: t for t in tools}
    faults = [_fault(f) for f in data.get("faults") or [{"id": "clean"}]]
    server = _get(data, "mcp_server")
    scenario = Scenario(
        skill_path=(path.parent / data.get("skill", ".")).resolve(),
        tasks=[
            Task(
                id=t["id"],
                prompt=t["prompt"],
                checks=_checks(t.get("checks") or []),
                state_checks=_state_checks(_get(t, "state_checks")),
            )
            for t in data["tasks"]
        ],
        tools=tools,
        faults=faults,
        initial_state=dict(_get(data, "initial_state") or {}),
        failure_words=_get(data, "failure_words", DEFAULT_FAILURE_WORDS),
        success_words=_get(data, "success_words", DEFAULT_SUCCESS_WORDS),
        contract=_get(data, "output_contract"),
        status_field=_get(data, "status_field"),
        mcp_server=list(server) if server else None,
    )
    auto = _get(data, "auto_faults")
    if auto:
        from ajantik.faults import generate

        known = {f.id for f in scenario.faults}
        for f in generate(scenario, list(auto)):
            if f.id not in known:
                scenario.faults.append(f)
    for f in scenario.faults:
        validate_fault(f, by_name)
    return scenario
