"""Trial harness: run a skill on a task under one fault condition and record every call.

A manual tool loop (not the SDK tool runner) because each call's usage is metered against a hard
budget before the next one starts, and tool results pass through the fault layer.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import anthropic

from ajantik.bench import FaultyTools
from ajantik.identity import Identity
from ajantik.oracle import CURRENT, classify, right_behaviour
from ajantik.pricing import cost_usd, worst_case_call_usd
from ajantik.scenario import Fault, Scenario, Task
from ajantik.skill import Skill

CHARS_PER_TOKEN_FLOOR = 2  # conservative: real text averages more chars per token


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetGuard:
    cap_usd: float
    spent_usd: float = 0.0

    def ensure(self, worst_case_usd: float) -> None:
        if self.spent_usd + worst_case_usd > self.cap_usd:
            raise BudgetExceeded(
                f"The next call could cost up to ${worst_case_usd:.3f}; "
                f"spent ${self.spent_usd:.3f}, cap ${self.cap_usd:.2f}."
            )

    def add(self, usd: float) -> None:
        self.spent_usd += usd


@dataclass
class TrialResult:
    identity: str
    task: str
    fault: str
    rep: int
    stop: str  # end_turn | max_tokens | refusal | turn_limit | budget | api_error
    success: bool | None  # None: the trial did not finish, so it is not evidence
    failed_checks: list[str] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    final_text: str = ""
    error: str = ""
    # Whether the declared fault actually fired. None: the bench cannot say.
    fault_fired: bool | None = None
    started_at: str = ""
    # success | honest_failure | silent_wrong | unclear | unfinished ("" = the trial did not finish)
    outcome: str = ""
    right: bool | None = None  # right behaviour: success if possible, otherwise an honest failure
    oracle: int = 0  # oracle version that produced outcome/right
    module: str = ""
    final_state: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


ORACLE_VERSION = CURRENT


def system_prompt(skill: Skill) -> str:
    extra = ""
    if skill.extra_files():
        extra = (
            "\nThe skill bundles these files, readable with the read_skill_file tool: "
            + ", ".join(skill.extra_files())
            + "\n"
        )
    if not skill.co:  # keep the single-skill prompt byte-identical for existing identities
        return (
            "You are an agent completing a task for a user. The following skill is installed; "
            "follow it when it applies.\n\n"
            f'<skill name="{skill.name}" description="{skill.description}">\n'
            f"{skill.body}\n</skill>\n{extra}"
        )
    blocks = "\n".join(
        f'<skill name="{s.name}" description="{s.description}">\n{s.body}\n</skill>'
        for s in [skill, *skill.co]
    )
    return (
        "You are an agent completing a task for a user. The following skills are installed; "
        "follow each one when it applies. Files of the other skills are prefixed with the skill "
        f"name.\n\n{blocks}\n{extra}"
    )


def _usage(u: Any) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


def _chars(content: Any) -> int:
    return len(json.dumps(content, default=str, ensure_ascii=False))


def run_trial(
    client: Any,
    skill: Skill,
    scenario: Scenario,
    task: Task,
    fault: Fault,
    identity: Identity,
    rep: int,
    guard: BudgetGuard,
    max_tokens: int = 4000,
    max_turns: int = 8,
    tools: Any = None,
) -> TrialResult:
    """`tools` lets a caller supply another bench with the same contract --
    `definitions()`, `execute()`, `.state`, `.session_dead` -- so the same scenario
    can run against a real MCP server (`ajantik.mcp.MCPTools`). None means the
    scenario's own simulated tools, which is what every existing caller gets."""
    result = TrialResult(
        identity=identity.id,
        task=task.id,
        fault=fault.id,
        rep=rep,
        stop="turn_limit",
        success=None,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    tools = tools or FaultyTools(skill, scenario.tools, fault, scenario.initial_state)
    system = [{"type": "text", "text": system_prompt(skill), "cache_control": {"type": "ephemeral"}}]
    tool_defs = tools.definitions()
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    prompt_tokens_est = (_chars(system) + _chars(tool_defs) + _chars(messages)) // CHARS_PER_TOKEN_FLOOR
    t0 = time.monotonic()
    response = None

    try:
        for _ in range(max_turns):
            guard.ensure(worst_case_call_usd(identity.model, prompt_tokens_est, max_tokens))
            try:
                response = client.messages.create(
                    model=identity.model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tool_defs,
                    messages=messages,
                    thinking={"type": "adaptive"},
                    output_config={"effort": identity.effort},
                )
            except anthropic.BadRequestError:
                raise
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                result.stop, result.error = "api_error", f"{type(e).__name__}: {e}"
                break

            usage = _usage(response.usage)
            call_cost = cost_usd(identity.model, usage)
            guard.add(call_cost)
            result.cost_usd += call_cost
            result.turns += 1
            for k, v in usage.items():
                setattr(result, k, getattr(result, k) + v)
            prompt_tokens_est = (
                usage["input_tokens"]
                + usage["cache_creation_input_tokens"]
                + usage["cache_read_input_tokens"]
                + usage["output_tokens"]
            )

            if response.stop_reason != "tool_use":
                result.stop = response.stop_reason or "unknown"
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result.tool_calls += 1
                text, is_error = tools.execute(block.name, block.input or {})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}
                )
            messages.append({"role": "user", "content": tool_results})
            prompt_tokens_est += _chars(tool_results) // CHARS_PER_TOKEN_FLOOR
    except BudgetExceeded as e:
        # The caller sees stop == "budget" and ends the session; the partial cost is kept.
        result.stop, result.error = "budget", str(e)

    result.seconds = round(time.monotonic() - t0, 2)
    if response is not None:
        result.final_text = "".join(b.text for b in response.content if b.type == "text")[:4000]
    result.module = fault.module
    result.final_state = dict(tools.state)
    result.fault_fired = getattr(tools, "fault_fired", None)
    if result.stop in ("end_turn", "max_tokens", "refusal", "turn_limit"):
        result.failed_checks = _failed_checks(scenario, task, fault, result.final_text, tools.state)
        # A fault that makes the task impossible settles success by construction: no set of
        # checks can make it true. Scenarios used to encode this with a never-matching regex
        # (`checks: - regex: '(?!)'`); stating it here means a condition whose right behaviour
        # is "say you could not do it" needs no task-specific checks at all.
        result.success = (result.stop == "end_turn" and not result.failed_checks
                          and fault.success_possible)
    result.outcome = classify(
        result.success, result.stop, result.final_text, scenario.failure_words,
        scenario.success_words, scenario.contract, status_field=scenario.status_field,
    )
    result.right = right_behaviour(result.outcome, fault.success_possible)
    result.oracle = ORACLE_VERSION
    return result


def _failed_checks(scenario, task, fault, text: str, state: dict) -> list[str]:
    checks = fault.checks if fault.checks is not None else task.checks
    state_checks = fault.state_checks if fault.state_checks is not None else task.state_checks
    failed = [c.label() for c in checks if not c.passes(text)]
    failed += [c.label() for c in state_checks if not c.passes(state)]
    if fault.relation:  # metamorphic: relate to the clean answer instead of an absolute truth
        from ajantik.anomalies import judge as metamorphic

        ok = metamorphic(fault.relation, fault.marker, not failed, text, scenario.contract)
        failed = [] if ok else [f"metamorphic: {fault.relation}"]
    return failed


def rescore(scenario, trial: dict) -> tuple[list[str], bool]:
    """Re-run the scenario's current checks on a stored trial (no API call)."""
    task = next(t for t in scenario.tasks if t.id == trial["task"])
    fault = next(f for f in scenario.faults if f.id == trial["fault"])
    failed = _failed_checks(scenario, task, fault, trial["final_text"], trial.get("final_state") or {})
    return failed, trial["stop"] == "end_turn" and not failed
