"""The round runner: trial isolation, substitution, and refusing to guess."""

import json
import sys
from pathlib import Path

import pytest

from ajantik.rounds import (
    MANIFEST,
    RECORDS_DIR,
    SANDBOX,
    TASK,
    WALL,
    main,
    run_round,
    wall_command,
)

ROOT = Path(__file__).parent.parent
EX = ROOT / "examples" / "intake-form" / "scenario.yaml"
STAND_IN = ROOT / "tests" / "stand_in_mcp_agent_en.py"
PHANTOM = "phantom-success:set_field"
FULL_FORM = {"product_name": "AURORA DESK LAMP", "country": "BG"}


def agent(style="blind", extra=()):
    return [sys.executable, str(STAND_IN), "--wall", WALL, "--task", TASK, "--style", style,
            *extra]


def final_state(record: Path) -> dict:
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    return next(r for r in rows if r["event"] == "session_end")["final_state"]


def test_wall_command_names_the_fault_and_the_record(tmp_path):
    cmd = wall_command(EX, PHANTOM, tmp_path / "r.jsonl")
    assert "--fault" in cmd and PHANTOM in cmd
    assert "--record" in cmd and str(tmp_path / "r.jsonl") in cmd
    assert "ajantik.wall" in cmd


def test_dry_run_substitutes_and_runs_nothing(tmp_path, capsys):
    m = run_round(EX, tmp_path, agent(), only=[PHANTOM], dry_run=True)
    printed = capsys.readouterr().out
    assert PHANTOM in printed and "Copy the product name" in printed
    assert WALL not in printed and TASK not in printed
    assert not (tmp_path / MANIFEST).exists()
    assert len(m["trials"]) == 1


def test_every_trial_gets_its_own_wall_and_directory(tmp_path):
    """Sharing either lets one trial's leftovers decide the next one's verdict."""
    m = run_round(EX, tmp_path, agent(), only=[PHANTOM, "clean"], reps=2)
    assert len(m["trials"]) == 4
    assert len({t["sandbox"] for t in m["trials"]}) == 4
    assert len({t["record"] for t in m["trials"]}) == 4
    assert all((tmp_path / t["record"]).exists() for t in m["trials"])


def test_new_rounds_keep_their_records_under_records(tmp_path):
    m = run_round(EX, tmp_path, agent(), only=["clean"])
    assert m["trials"][0]["record"].startswith(f"{RECORDS_DIR}/")
    from ajantik import legacy

    assert not any((tmp_path / d).exists() for d in legacy.RECORD_DIRS)


def test_a_full_round_covers_every_fault_in_the_scenario(tmp_path):
    m = run_round(EX, tmp_path, agent())
    assert len(m["faults"]) == 14
    assert len(m["trials"]) == 14
    assert all(t["exit_code"] == 0 and not t["timed_out"] for t in m["trials"])
    assert json.loads((tmp_path / MANIFEST).read_text())["format"] == "ajantik.round.v1"


def test_the_blind_stand_in_loses_the_phantom_write(tmp_path):
    """A sanity check on the whole chain: the fault reaches the agent and shows up
    in the recorded final state."""
    run_round(EX, tmp_path, agent("blind"), only=[PHANTOM, "clean"])
    states = {record.stem.rsplit("-", 1)[0]: final_state(record)
              for record in (tmp_path / RECORDS_DIR).glob("*.jsonl")}
    assert states["clean"] == FULL_FORM
    assert states["phantom-success_set_field"] == {"country": "BG"}


def test_verifying_stand_in_recovers_it(tmp_path):
    run_round(EX, tmp_path, agent("verifying"), only=[PHANTOM])
    assert final_state(next((tmp_path / RECORDS_DIR).glob("*.jsonl"))) == FULL_FORM


def test_agent_stdout_and_stderr_are_kept(tmp_path):
    run_round(EX, tmp_path, agent(), only=["clean"])
    out = (tmp_path / RECORDS_DIR / "clean-001.agent-stdout.txt").read_text()
    assert "Done!" in out
    assert (tmp_path / RECORDS_DIR / "clean-001.agent-stderr.txt").exists()


def test_a_timeout_leaves_no_session_end_so_it_cannot_score_as_behaviour(tmp_path):
    # The tokens go in as arguments, not into the code: the task text has an apostrophe.
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)", WALL, TASK]
    m = run_round(EX, tmp_path, sleeper, only=["clean"], timeout_s=1)
    assert m["trials"][0]["timed_out"] is True and m["trials"][0]["exit_code"] is None
    record = tmp_path / m["trials"][0]["record"]
    assert not record.exists() or "session_end" not in record.read_text()


def test_a_relative_agent_path_is_explained_not_swallowed(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_round(EX, tmp_path, ["./not-here", WALL, TASK], only=["clean"])
    assert "absolute paths" in str(exc.value)


def test_unknown_fault_is_refused_with_the_known_ones(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_round(EX, tmp_path, agent(), only=["no-such-fault"])
    assert PHANTOM in str(exc.value)


def test_cli_refuses_a_command_that_never_reaches_the_wall(tmp_path):
    for args, expect in (
        ([sys.executable, "x.py", "--task", TASK], "{wall}"),
        ([sys.executable, "x.py", "--wall", WALL], "{task}"),
    ):
        with pytest.raises(SystemExit) as exc:
            main(["--scenario", str(EX), "--out", str(tmp_path), "--dry-run", "--", *args])
        assert expect in str(exc.value)


def test_cli_refuses_an_empty_agent_command(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--scenario", str(EX), "--out", str(tmp_path), "--dry-run"])
    assert "no agent command" in str(exc.value)


def test_sandbox_token_is_substituted_when_used(tmp_path, capsys):
    run_round(EX, tmp_path, [sys.executable, "x.py", WALL, TASK, SANDBOX],
              only=["clean"], dry_run=True)
    assert SANDBOX not in capsys.readouterr().out


def test_a_relative_output_directory_still_reaches_the_wall(tmp_path, monkeypatch):
    """The wall runs in the trial's fresh directory, so a relative record path resolved
    against it: the wall crashed, the agent had no tools, and a whole 40-trial round was
    lost. The runner makes the output directory absolute itself."""
    monkeypatch.chdir(tmp_path)
    m = run_round(EX, Path("rel-out"), agent(), only=[PHANTOM])
    rec = tmp_path / "rel-out" / m["trials"][0]["record"]
    assert rec.exists() and "session_end" in rec.read_text()
