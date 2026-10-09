"""The synthetic portal pilot behaves like the portal it stands in for."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PILOT = Path(__file__).parent.parent / "examples" / "portal-pilot"


@pytest.fixture
def run(tmp_path, monkeypatch):
    env = {"PORTAL_HOME": str(tmp_path / "portal"), "PORTAL_IDLE_MIN": "15", "PATH": "/usr/bin:/bin"}

    def call(script, *args, idle=None):
        e = {**env, **({"PORTAL_IDLE_MIN": str(idle)} if idle is not None else {})}
        done = subprocess.run([sys.executable, str(PILOT / "tools" / f"{script}.py"), *args],
                              cwd=PILOT, env=e, capture_output=True, text=True, check=True)
        lines = done.stdout.strip().splitlines()
        assert len(lines) == 1, done.stdout  # one JSON line, always
        return json.loads(lines[0])
    return call


def test_nothing_works_before_the_operator_logs_in(run):
    assert run("status")["stop"] == "not_logged_in"
    assert run("open_record", "--record", "R-101")["result"] == "stop"


def test_a_full_record_by_script(run):
    assert run("login", "--account", "ACME")["account"] == "ACME"
    opened = run("open_record", "--record", "R-102")
    assert opened["filled"] == [] and opened["product"] == "Borealis Cleaner"
    saved = run("fill", "--record", "R-102", "--section", "2",
                "--rows", json.dumps([{"part": "a"}, {"part": "b"}, {"part": "c"}]))
    assert (saved["saved"], saved["accepted"]) == (3, 3)
    up = run("upload", "--record", "R-102", "--file", "docs-in/R-102.pdf")
    assert up["bytes"] > 1000
    counts = run("scan", "--record", "R-102")["counts"]
    assert counts["2"] == 3 and counts["7"] == 1 and counts["1"] == 0


def test_accounts_are_separate(run):
    run("login", "--account", "GLOBEX")
    assert run("open_record", "--record", "R-101")["product"] == "Globex Primer"
    run("login", "--account", "ACME")
    assert run("open_record", "--record", "R-101")["product"] == "Aurora Desk Lamp"


def test_an_idle_session_ends(run):
    run("login", "--account", "ACME")
    assert run("status", idle=0)["stop"] == "session_expired"
    assert run("status")["stop"] == "not_logged_in"  # it stays ended


def test_an_empty_upload_still_says_ok(run, tmp_path):
    run("login", "--account", "ACME")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    up = run("upload", "--record", "R-101", "--file", str(empty))
    assert up["result"] == "ok" and up["bytes"] == 0  # only the skill's own rule catches it


def test_the_skill_states_its_expected_counts_and_report_format():
    skill = (PILOT / ".claude" / "skills" / "register-products" / "SKILL.md").read_text()
    assert "VERIFIED 8/8" in skill and "accepted" in skill and "bytes" in skill
