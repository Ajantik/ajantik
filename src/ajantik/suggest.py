"""Rule suggestion: turn failed trials into a hardened SKILL.md.

The suggester sees the failing (non-held-out) conditions only: what the tool returned, what the
agent wrote and the correct behaviour in plain words. It never sees the checks or the held-out
conditions, so a later run on held-out faults tells whether the rules generalize.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from ajantik.pricing import cost_usd, worst_case_call_usd
from ajantik.scenario import Scenario
from ajantik.skill import Skill
from ajantik.trial import BudgetGuard

MAX_TOKENS = 5000


# The field names are the JSON schema the suggester fills and are read by `ajantik suggest`
# in cli.py: rule text, reason, the conditions it answers; the rules and the new SKILL.md.
class Rule(BaseModel):
    rule: str
    reason: str
    conditions: list[str]


class Suggestion(BaseModel):
    rules: list[Rule]
    new_skill_md: str


INSTRUCTIONS = """A Claude skill produced a wrong result under some test conditions in which tool data was corrupted.
Your task is to harden the SKILL.md file with as few changes as possible.

- Rules must be general: do not write the specific order numbers, amounts, dates or currencies
  from these examples into a rule. They should also cover cases of the same kind you have not seen here.
- Do not break the conditions that pass. Do not change the output format (especially the format of the last line).
- Keep name and description in the header (between ---) exactly as they are.
- Write the whole file in the new_skill_md field."""


def build_prompt(skill: Skill, scenario: Scenario, trials: list[dict]) -> str:
    by_fault: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if t["success"] is not None:
            by_fault[t["fault"]].append(t)
    visible = {f.id: f for f in scenario.faults if not f.held_out}
    tool_response = {t.name: t.response for t in scenario.tools}

    parts = [INSTRUCTIONS, "\n## Current SKILL.md\n", skill.files["SKILL.md"].decode("utf-8")]
    parts.append("\n## Task\n" + "\n".join(t.prompt for t in scenario.tasks))
    passing, failing = [], []
    for fid, ts in by_fault.items():
        if fid not in visible:
            continue
        (failing if any(not t["success"] for t in ts) else passing).append(fid)
    for fid in failing:
        fault, ts = visible[fid], by_fault[fid]
        bad = next(t for t in ts if not t["success"])
        data = fault.response if fault.kind == "replace" else tool_response.get(fault.tool or "", "")
        parts += [
            f"\n## Failing condition: {fid} ({sum(not t['success'] for t in ts)}/{len(ts)} trials failed)",
            f"Right behaviour: {fault.description or 'not specified'}",
            f"Data the tool returned:\n{data}",
            f"The agent's output (last part):\n{bad['final_text'][-900:]}",
        ]
    parts.append("\n## Passing conditions (must not break)\n" + (", ".join(passing) or "none"))
    return "\n".join(parts)


def suggest(
    client: Any, skill: Skill, scenario: Scenario, trials: list[dict], model: str, guard: BudgetGuard
) -> tuple[Suggestion, float]:
    prompt = build_prompt(skill, scenario, trials)
    guard.ensure(worst_case_call_usd(model, len(prompt) // 2, MAX_TOKENS))
    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        output_format=Suggestion,
    )
    usage = {
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }
    cost = cost_usd(model, usage)
    guard.add(cost)
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"No suggestion received (stop_reason={response.stop_reason}).")
    return response.parsed_output, cost
