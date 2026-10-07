"""The no-key demo shows the lab's whole point: checking your work is what separates the two."""

import json
from pathlib import Path

import pytest

from ajantik.demo import run
from ajantik.scenario import load_scenario
from ajantik.verdicts import verdicts

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("name", ["support-ticket", "calendar-booking", "crm-update"])
def test_blind_ends_silent_wrong_and_verifying_ends_correct(tmp_path, name):
    scen = EXAMPLES / name / "scenario.yaml"
    report = run(scen, tmp_path, reps=1)
    assert report.exists() and "Ajantik Demo" in report.read_text()
    for style, expect in (("blind", {"silent_wrong"}), ("verifying", {"correct"})):
        trials = verdicts(scen, {style: tmp_path / style}, judge_file=None)["trials"]
        assert trials and {t["verdict"] for t in trials} == expect


@pytest.mark.parametrize("name", ["support-ticket", "calendar-booking", "crm-update"])
def test_every_example_needs_its_source(name):
    """A fault on the source must matter: the values are in the source, not in the task."""
    scen = load_scenario(EXAMPLES / name / "scenario.yaml")
    source = next(t for t in scen.tools if t.effect == "none")
    for check in scen.tasks[0].state_checks:
        assert str(check.equals) not in scen.tasks[0].prompt
        assert str(check.equals) in json.dumps(json.loads(source.response)) or \
            str(check.equals).lower() in source.response.lower()


def test_the_packaged_demo_scenario_is_the_support_ticket_example():
    """The demo ships its own copy so it runs from a pip install; it must not drift."""
    from ajantik.demo import DEFAULT_SCENARIO

    packaged = DEFAULT_SCENARIO.read_text().replace("skill: .\n", "skill: v1\n")
    assert packaged == (EXAMPLES / "support-ticket" / "scenario.yaml").read_text()
