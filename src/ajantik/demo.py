"""A thirty-second demo with no API key: two scripted agents, one round, one report.

    ajantik demo --out demo/

Both agents are scripts, not models, and both know the right values in advance. The only
difference between them is whether they check their work:

  blind      writes each value, ignores what the tool answers, says "Done"
  verifying  retries a failed write once, reads every value back, rewrites a mismatch,
             and says what it could not save

Against a tool that says "saved" and stores nothing, the blind agent ends silent wrong and
the verifying one ends correct. That is the whole idea of the lab, before any model is
involved. The agents run in-process (`ajantik.adapter.Wall`), the same way an agent built in
code would.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Shipped inside the package so `pip install ajantik && ajantik demo` works with no checkout.
# A copy of examples/support-ticket (a test keeps the two identical).
DEFAULT_SCENARIO = Path(__file__).with_name("demo_scenario.yaml")
DEMO_FAULTS = ("phantom-success", "transient-error")


def play(style: str) -> str:
    """Run one scripted agent against the wall the round runner set up. Returns its message."""
    from ajantik.adapter import Wall

    with Wall.from_env() as wall:
        scen = wall.scenario
        source = next((t for t in scen.tools if t.effect == "none"), None)
        writer = next(t for t in scen.tools if t.effect == "write")
        reader = next((t for t in scen.tools if t.effect == "read"), None)
        if source:
            wall.call(source.name, {})
        failed = []
        for check in scen.tasks[0].state_checks:
            args = {writer.key_field: check.key, writer.value_field: check.equals}
            _, is_error = wall.call(writer.name, args)
            if style == "blind":
                continue
            if is_error:
                _, is_error = wall.call(writer.name, args)
            if reader:
                text, _ = wall.call(reader.name, {reader.key_field: check.key})
                try:
                    stored = json.loads(text).get(reader.value_field)
                except ValueError:
                    stored = None
                if stored != check.equals:
                    wall.call(writer.name, args)
                    text, _ = wall.call(reader.name, {reader.key_field: check.key})
                    if json.loads(text).get(reader.value_field) != check.equals:
                        failed.append(check.key)
    if failed:
        return (f"I could not save {', '.join(failed)}: the system reported success but "
                "reading it back shows nothing stored. Please check before relying on it.")
    return "Done! All fields were saved."


def run(scenario: Path, out: Path, reps: int = 3) -> Path:
    from ajantik.page import page_data, render
    from ajantik.rounds import TASK, run_round
    from ajantik.scenario import load_scenario

    scen = load_scenario(scenario)
    faults = [f.id for f in scen.faults
              if f.id.split(":")[0] in DEMO_FAULTS and f.tool in
              {t.name for t in scen.tools if t.effect == "write"}]
    rounds = {}
    for style in ("blind", "verifying"):
        cmd = [sys.executable, "-m", "ajantik.demo", "agent", "--style", style, "--task", TASK]
        run_round(scenario, out / style, cmd, reps=reps, only=faults, in_process=True)
        rounds[f"{style} (scripted)"] = out / style
    report = out / "report.html"
    report.write_text(render(page_data(scenario, rounds), "Ajantik Demo"), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ajantik.demo")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agent", help="one scripted agent (started by the round runner)")
    a.add_argument("--style", choices=["blind", "verifying"], required=True)
    a.add_argument("--task", required=True)
    args = p.parse_args(argv)
    if args.cmd == "agent":
        print(play(args.style))
    return 0


if __name__ == "__main__":
    sys.exit(main())
