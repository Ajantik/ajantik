"""Where the judge was measured, per-agent extraction, and the warning where it was not."""

import json
import stat
import sys
from pathlib import Path

from ajantik import judge
from ajantik.agents import agent_name, cost_usd, profile_for
from ajantik.rounds import MCP_CONFIG, TASK, mcp_config, run_round
from ajantik.verdicts import verdicts

ROOT = Path(__file__).parent.parent
EN = ROOT / "examples" / "intake-form" / "scenario.yaml"
PHANTOM = "phantom-success:set_field"


def fake_claude(tmp_path: Path) -> Path:
    """An executable named `claude`, so the round records the agent as Claude Code."""
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'stand_in_claude.py'} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_goose_is_measured_in_both_languages_and_others_are_not(tmp_path):
    none = tmp_path / "absent.yaml"
    assert judge.status("goose", "en", 5, none)["measured"] is True
    assert judge.status("goose", "tr", 5, none)["kappa"] == 0.66
    assert judge.status("claude-code", "en", 5, none)["measured"] is True
    s = judge.status("claude-code", "tr", 5, none)
    assert s["measured"] is False and "judge-sample" in s["warning"]


def test_an_english_task_is_detected_as_english():
    """The Turkish side of this check lives in tests/lab/test_judge_tr.py."""
    assert judge.language_of("Copy the product name into the form.") == "en"


def test_false_alarms_and_misses_get_different_notes(tmp_path):
    assert "overstate" in judge.status("goose", "en", 5, tmp_path / "x")["note"]
    m = judge.Measurement("x", "en", 5, 10, 8, 7, 0.5, 0.2, 0.8, 3, 1, 0, 4, "s", "d")
    judge.register(m, tmp_path / "j.yaml")
    assert "MISSED" in judge.status("x", "en", 5, tmp_path / "j.yaml")["note"]


def test_registered_measurements_are_read_back(tmp_path):
    m = judge.Measurement("claude-code", "en", 5, 20, 16, 14, 0.7, 0.5, 0.85, 7, 0, 2, 7,
                          "local", "2026-10-06")
    judge.register(m, tmp_path / "j.yaml")
    assert judge.status("claude-code", "en", 5, tmp_path / "j.yaml")["measured"] is True


def test_claude_code_closing_message_comes_from_its_json(tmp_path):
    out = json.dumps({"type": "result", "result": "Saved both.", "total_cost_usd": 0.02})
    p = profile_for(["/usr/local/bin/claude", "-p"])
    assert p.name == "claude-code" and p.extract("noise\n" + out, []) == "Saved both."
    assert cost_usd(out) == 0.02
    assert agent_name(["/opt/homebrew/bin/goose"]) == "goose"
    assert profile_for(["python3"]).name == "generic"
    assert agent_name(["/usr/bin/python3.12", "-u", "agents/my_agent.py", "--x"]) == "my_agent"
    assert agent_name(["npx", "-y", "some-agent"]) == "some-agent"


def test_mcp_config_is_the_form_claude_code_takes():
    cfg = json.loads(mcp_config(["python", "-m", "ajantik.wall", "--scenario", "s.yaml"]))
    assert cfg == {"mcpServers": {"ajantik": {"command": "python",
                                              "args": ["-m", "ajantik.wall", "--scenario", "s.yaml"]}}}


def test_a_claude_code_round_reads_its_json_and_finds_its_measurement(tmp_path, monkeypatch):
    """A Claude Code round: the claim is read from its JSON, and the document says the
    judge was never measured on it, in the limitations a report must publish."""
    exe = fake_claude(tmp_path)
    run_round(EN, tmp_path / "r", [str(exe), "-p", TASK, "--mcp-config", MCP_CONFIG,
                                   "--output-format", "json"], only=[PHANTOM])
    doc = verdicts(EN, {"m": tmp_path / "r"}, judge_file=tmp_path / "absent.yaml")
    t = doc["trials"][0]
    assert (t["verdict"], t["claim"]) == ("silent_wrong", "silent_wrong")
    assert doc["judge"][0]["agent"] == "claude-code" and doc["judge"][0]["measured"] is True
    other = judge.status("claude-code", "tr", 5, tmp_path / "absent.yaml")
    assert "not been measured on claude-code in tr" in other["warning"]


def test_an_agent_that_never_authenticated_gets_no_verdict(tmp_path):
    """Seen on the first real Claude Code trial: an expired login. The wall still recorded a
    session with the world wrong, and 'Failed to authenticate' reads like an honest report.
    Scoring it would turn our infrastructure failure into the agent's behaviour."""
    exe = fake_claude(tmp_path)
    run_round(EN, tmp_path / "r", [str(exe), "-p", TASK, "--mcp-config", MCP_CONFIG,
                                   "--output-format", "json"], only=[PHANTOM])
    from ajantik import legacy

    out = next(p for d in ("records", *legacy.RECORD_DIRS)
               for p in (tmp_path / "r" / d).glob("*.agent-stdout.txt"))
    out.write_text(json.dumps({"type": "result", "is_error": True, "num_turns": 1,
                               "terminal_reason": "api_error", "usage": {"output_tokens": 0},
                               "result": "Failed to authenticate: OAuth session expired"}))
    t = verdicts(EN, {"m": tmp_path / "r"}, judge_file=tmp_path / "absent.yaml")["trials"][0]
    assert t["verdict"] == "no_verdict" and "authenticate" in t["reason"]


def test_a_perfect_agreement_has_no_kappa_interval_but_says_why():
    """Seen measuring Claude Code: 19/19 matched and the bootstrap printed '1.0-1.0'. With no
    disagreement every resample gives 1; that is not an interval. Report the raw agreement's."""
    m = judge.Measurement("x", "en", 5, 19, 19, 19, 1.0, None, None, 9, 0, 0, 10, "s", "d")
    text = judge.interval_text(m)
    assert "no disagreement" in text and "19/19" in text and "1.0-1.0" not in text
    assert judge.interval_text(judge.BUILTIN[1]) == "80% interval 0.3-0.67"
