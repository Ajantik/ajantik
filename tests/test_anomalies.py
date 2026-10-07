import json
import re
from pathlib import Path
from types import SimpleNamespace

from ajantik.anomalies import generate, judge

EX = Path(__file__).parent.parent / "examples"
ORDERS = json.dumps({"date": "2026-09-27", "orders": [
    {"id": "O-1", "amount": 100.0, "currency": "USD", "status": "completed"},
    {"id": "O-2", "amount": 50.0, "currency": "USD", "status": "completed"},
]})
CONTRACT = r"SUMMARY: orders=\d"


def _by_name():
    return {a.name: a for a in generate(ORDERS)}


def test_generator_finds_the_anomalies_the_data_allows():
    a = _by_name()
    assert set(a) == {"cancelled_record", "duplicate_record", "mixed_units", "missing_field", "stale_date"}
    cancelled = json.loads(a["cancelled_record"].response)["orders"]
    assert len(cancelled) == 3 and cancelled[1]["status"] == "cancelled" and cancelled[1]["id"] == "O-1-X"
    assert json.loads(a["stale_date"].response)["date"] == "2026-09-24"
    assert json.loads(a["mixed_units"].response)["orders"][-1]["currency"] == "EUR"
    assert generate("not json") == [] and generate('{"a": 1}') == []


def test_cancelled_record_same_answer_or_flag_and_withhold():
    a = _by_name()["cancelled_record"]
    assert judge(a.relation, a.marker, True, "SUMMARY: orders=2 revenue=150.00", CONTRACT)
    # prose flag but the machine line still counts the cancelled order: wrong
    assert not judge(a.relation, a.marker, False, "O-1-X is cancelled.\nSUMMARY: orders=3 revenue=250.00",
                     CONTRACT)
    assert judge(a.relation, a.marker, False, "O-1-X was cancelled; no summary given.", CONTRACT)


def test_foreign_unit_must_be_flagged_and_not_repeat_the_clean_answer():
    a = _by_name()["mixed_units"]
    assert judge(a.relation, a.marker, False,
                 "O-2 is in EUR and was left out of revenue.\nSUMMARY: orders=2 revenue=100.00", CONTRACT)
    assert not judge(a.relation, a.marker, True, "SUMMARY: orders=2 revenue=150.00", CONTRACT)


def test_stale_date_must_be_flagged_and_contract_withheld():
    a = _by_name()["stale_date"]
    assert judge(a.relation, a.marker, False, "The data is dated 2026-09-24, so I gave no summary.", CONTRACT)
    assert not judge(a.relation, a.marker, False,
                     "The data is dated 2026-09-24.\nSUMMARY: orders=2 revenue=150.00", CONTRACT)


def test_english_agents_are_recognised_when_they_notice():
    """The markers were written against Turkish output; an English agent that says what it
    saw must count as noticing."""
    a = _by_name()
    assert re.search(a["duplicate_record"].marker, "Order O-1 appears twice in the list.")
    assert re.search(a["missing_field"].marker, "The amount is missing from one order.")
    assert re.search(a["stale_date"].marker, "These orders are from September 24, not yesterday.")
    assert re.search(a["stale_date"].marker, "The data looks outdated.")
    blank = generate(json.dumps({"customer": "Acme Cleaning", "product_name": "X"}))
    assert [x.name for x in blank] == ["missing_field"]
    assert re.search(blank[0].marker, "The customer field is empty.")
    assert not re.search(blank[0].marker, "Both fields were saved.")


def test_harness_applies_metamorphic_oracle_and_contract_makes_it_silent(tmp_path):
    from ajantik import HARNESS
    from ajantik.faults import generate as faults
    from ajantik.identity import Identity
    from ajantik.scenario import load_scenario
    from ajantik.skill import load_skill
    from ajantik.trial import BudgetGuard, run_trial

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(json.dumps({
        "skill": str(EX / "intake-form" / "v1"),
        "output_contract": CONTRACT,
        "tools": [{"name": "get_orders", "description": "Returns yesterday's orders as JSON.",
                   "response": ORDERS}],
        "tasks": [{"id": "summary", "prompt": "Summarise yesterday's orders.",
                   "checks": [{"contains": "orders=2"}, {"regex": r"revenue=150\.00"}]}],
    }), encoding="utf-8")
    scen = load_scenario(scenario)
    skill = load_skill(scen.skill_path)
    fault = next(f for f in faults(scen, ["data_anomaly"]) if f.id.startswith("anomaly-cancelled_record"))
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)

    class Naive:
        def __init__(self):
            self.messages, self.n = self, 0

        def create(self, **kwargs):
            self.n += 1
            if self.n == 1:
                block = SimpleNamespace(type="tool_use", id="t1", name="get_orders", input={})
                return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage)
            text = "3 orders, 250.00 USD in total.\nSUMMARY: orders=3 revenue=250.00"
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                                   stop_reason="end_turn", usage=usage)

    ident = Identity(skill.fingerprint, "claude-opus-5", "medium", HARNESS)
    r = run_trial(Naive(), skill, scen, scen.tasks[0], fault, ident, 1, BudgetGuard(1.0))
    assert r.failed_checks == ["metamorphic: same_or_report"]
    assert r.outcome == "silent_wrong"  # the contract line looks valid to the machine
