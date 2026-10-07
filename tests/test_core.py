import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ajantik import HARNESS
from ajantik.estimate import summarize_cost, summarize_rate, widening, wilson
from ajantik.identity import Identity, recipe_files, recipe_fingerprint
from ajantik.report import build_report
from ajantik.scenario import Check, Fault, StateCheck, load_scenario
from ajantik.skill import load_skill
from ajantik.trial import BudgetGuard, run_trial

EXAMPLE = Path(__file__).parent.parent / "examples" / "intake-form"
RIGHT_FIELDS = {"product_name": "AURORA DESK LAMP", "country": "BG"}
ANSWER = "Saved product_name and country into the form.\nSTATUS: ok"


def _write_skill(root: Path, body: str = "Do it.") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(f"---\nname: s\ndescription: d\n---\n{body}\n")
    (root / "notes.txt").write_text("a\r\nb\r\n")
    return root


def test_fingerprint_same_for_dir_and_zip_and_line_endings(tmp_path):
    d = _write_skill(tmp_path / "skill")
    z = tmp_path / "skill.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(d / "SKILL.md", "skill/SKILL.md")
        zf.writestr("skill/notes.txt", "a\nb\n")
        zf.writestr("__MACOSX/skill/._SKILL.md", "junk")
    assert recipe_fingerprint(recipe_files(d)) == recipe_fingerprint(recipe_files(z))


def test_one_word_changes_identity(tmp_path):
    a = load_skill(_write_skill(tmp_path / "a", "Do it."))
    b = load_skill(_write_skill(tmp_path / "b", "Do it now."))
    assert a.fingerprint != b.fingerprint
    ia = Identity(a.fingerprint, "claude-opus-5", "medium", HARNESS)
    assert ia.id != Identity(a.fingerprint, "claude-sonnet-5", "medium", HARNESS).id


def test_rate_summary_rule_of_three():
    r = summarize_rate([True] * 5)
    assert r.max_failure_rate_95 == pytest.approx(0.6)
    low, high = wilson(5, 5)
    assert high == pytest.approx(1.0) and 0.6 < low < 1.0
    assert summarize_rate([True, False]).max_failure_rate_95 is None


def test_widening_and_mean_not_widened():
    assert widening(1) == 1.0
    assert widening(5) == pytest.approx(1.725)
    assert widening(2) == 3.0  # capped
    costs = [0.0258, 0.0264, 0.0273, 0.0390, 0.0623]
    s = summarize_cost(costs)
    assert s.mean == pytest.approx(np.mean(costs))
    assert s.p50 == pytest.approx(0.0273)
    assert s.p90 > np.percentile(costs, 90)  # band widened
    assert s.total_p10 < 1000 * s.mean < s.total_p90


def test_single_measurement_has_no_band():
    s = summarize_cost([0.03])
    assert s.p10 is None and s.p90 is None and s.p50 == 0.03


class FakeClient:
    """Scripted responses: first writes the form fields in one turn, then answers."""

    def __init__(self, answer: str, fields: dict[str, str] | None = None):
        self.answer = answer
        self.fields = RIGHT_FIELDS if fields is None else fields
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=0
        )
        last = kwargs["messages"][-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            blocks = [
                SimpleNamespace(type="tool_use", id=f"t{i}", name="set_field",
                                input={"field": k, "value": v})
                for i, (k, v) in enumerate(self.fields.items())
            ]
            return SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.answer)], stop_reason="end_turn", usage=usage
        )


def _setup():
    scen = load_scenario(EXAMPLE / "scenario.yaml")
    skill = load_skill(scen.skill_path)
    ident = Identity(skill.fingerprint, "claude-opus-5", "medium", HARNESS)
    return scen, skill, ident


def test_trial_success_and_cost():
    scen, skill, ident = _setup()
    client = FakeClient(ANSWER)
    guard = BudgetGuard(cap_usd=1.0)
    r = run_trial(client, skill, scen, scen.tasks[0], scen.faults[0], ident, 1, guard)
    assert r.success is True and r.turns == 2 and r.tool_calls == 2
    assert r.cost_usd == pytest.approx(2 * (1000 * 5 + 200 * 25) / 1e6)
    assert guard.spent_usd == pytest.approx(r.cost_usd)


def test_fault_is_injected_and_fault_checks_apply():
    scen, skill, ident = _setup()
    transient = next(f for f in scen.faults if f.id == "transient-error:set_field")
    client = FakeClient(ANSWER)
    run_trial(client, skill, scen, scen.tasks[0], transient, ident, 1, BudgetGuard(1.0))
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True and "503" in tool_result["content"]

    # The record has no country: the right answer changes, so the fault carries its own checks.
    missing = Fault(
        id="missing-country", kind="replace", tool="read_intake",
        response='{"customer": "Acme Cleaning", "product_name": "AURORA DESK LAMP"}',
        checks=[Check("contains", "country is missing")],
        state_checks=[StateCheck("product_name", "AURORA DESK LAMP"), StateCheck("country", None)],
    )
    invented = FakeClient(ANSWER)  # stores a country the record does not have
    r = run_trial(invented, skill, scen, scen.tasks[0], missing, ident, 1, BudgetGuard(1.0))
    assert r.success is False and any("country is missing" in c for c in r.failed_checks)
    assert any("state[country]" in c for c in r.failed_checks)


def test_budget_stops_before_overspending():
    scen, skill, ident = _setup()
    client = FakeClient("x")
    guard = BudgetGuard(cap_usd=0.05)  # below one worst-case call at max_tokens=4000
    r = run_trial(client, skill, scen, scen.tasks[0], scen.faults[0], ident, 1, guard)
    assert r.stop == "budget" and r.success is None and client.calls == []
    assert guard.spent_usd == 0


def test_report_says_not_measured_without_data():
    _, skill, ident = _setup()
    report, summary = build_report(ident, skill.name, [])
    assert "not measured" in report and summary["success_rate"] is None
    json.dumps(summary)


def test_rescore_uses_current_checks():
    from ajantik.trial import rescore

    scen = load_scenario(EXAMPLE / "scenario.yaml")
    base = {"task": "fill-form", "stop": "end_turn", "final_text": ANSWER, "fault": "clean"}
    assert rescore(scen, {**base, "final_state": RIGHT_FIELDS}) == ([], True)
    failed, success = rescore(scen, {**base, "final_state": {**RIGHT_FIELDS, "country": "GB"}})
    assert not success and failed


def test_combined_skill_has_own_identity_and_prompt(tmp_path):
    from ajantik.skill import combine
    from ajantik.trial import system_prompt

    main = load_skill(_write_skill(tmp_path / "a"))
    other_dir = tmp_path / "b"
    other_dir.mkdir()
    (other_dir / "SKILL.md").write_text("---\nname: other\ndescription: o\n---\nBe brief.\n")
    other = load_skill(other_dir)
    both = combine(main, [other])
    assert both.fingerprint != main.fingerprint
    assert "other/SKILL.md" in both.files and "notes.txt" in both.files
    prompt = system_prompt(both)
    assert prompt.count("<skill ") == 2 and "Be brief." in prompt
    # single-skill prompt unchanged, so old identities keep their meaning
    assert system_prompt(main).startswith("You are an agent completing a task for a user. The following skill is")
    assert combine(main, []) is main
