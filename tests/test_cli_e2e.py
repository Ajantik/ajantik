"""The whole self-serve path through the command line, with stand-in agents and $0 spent."""

import json
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from ajantik.cli import app

ROOT = Path(__file__).parent.parent
EN = ROOT / "examples" / "intake-form" / "scenario.yaml"
AGENT = ROOT / "tests" / "stand_in_mcp_agent_en.py"
runner = CliRunner()


def ok(args):
    res = runner.invoke(app, args, catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res.output


def test_record_round_verdicts_report_and_judge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # judge-agreement --register writes ./ajantik-judge.yaml
    scen = tmp_path / "scenario.yaml"
    real = f"{sys.executable} -m ajantik.wall --scenario {EN}"
    blind = [sys.executable, str(AGENT), "--wall", "{wall}", "--task", "{task}",
             "--style", "blind"]

    refused = runner.invoke(app, ["record", "--server", real, "--task", "Copy the fields.",
                                  "--out", str(scen), "--", *blind[:-1], "verifying"])
    assert refused.exit_code == 2 and "--allow-real-calls" in refused.output

    out = ok(["record", "--server", real, "--task", "Copy the fields.", "--out", str(scen),
              "--allow-real-calls", "--", *blind[:-1], "verifying"])
    assert "product_name == 'AURORA DESK LAMP'" in out

    ok(["round", "--scenario", str(scen), "--out", str(tmp_path / "r"), "--reps", "2",
        "--fault", "phantom-success:set_field", "--fault", "clean", "--", *blind])
    out = ok(["verdicts", "--scenario", str(scen), "--round", f"blind={tmp_path / 'r'}",
              "--out", str(tmp_path / "v.json")])
    assert "NOT MEASURED" in out and "stand_in_mcp_agent_en" in out  # never measured: flagged
    doc = json.loads((tmp_path / "v.json").read_text())
    assert {t["verdict"] for t in doc["trials"] if t["fault"].startswith("phantom")} == {"silent_wrong"}

    out = ok(["report", "--scenario", str(scen), "--round", f"blind={tmp_path / 'r'}",
              "--out", str(tmp_path / "report.html"), "--title", "Stand-in Crash Test"])
    page = (tmp_path / "report.html").read_text()
    assert "<title>Stand-in Crash Test</title>" in page and "phantom-success:set_field" in page

    ok(["judge-sample", "--scenario", str(scen), "--round", f"blind={tmp_path / 'r'}",
        "--out", str(tmp_path / "j"), "--size", "20"])
    form = yaml.safe_load((tmp_path / "j" / "form.yaml").read_text())
    for item in form["items"]:
        item["label"] = "silent"  # the stand-in claims success over an empty form
    (tmp_path / "j" / "form.yaml").write_text(yaml.safe_dump(form, allow_unicode=True))
    out = ok(["judge-agreement", "--dir", str(tmp_path / "j"), "--agent", "stand-in",
              "--register"])
    assert "Registered" in out
    assert yaml.safe_load((tmp_path / "ajantik-judge.yaml").read_text())[0]["agent"] == "stand-in"


def test_round_help_is_the_round_runners_own():
    res = runner.invoke(app, ["round", "--help"])
    assert "{wall_url}" in res.output and "--in-process" in res.output
