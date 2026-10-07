"""Run one round: every fault in a scenario, N repetitions, one agent we do not own.

A round is a sweep over (fault x repetition) for one (scenario, agent, model). Each
trial spawns a fresh wall (`ajantik.wall`) and runs the agent against it, so the
tool state cannot leak between trials. The agent also gets a fresh working directory
per trial, for the reason recorded in `cli.py`: sharing one directory let a trial
whose writes had all failed read the previous trial's leftovers and report success,
which is our error and not the agent's, and it invalidated two real runs.

The agent command is a template, so the round is agent-agnostic - Goose today, another
MCP client tomorrow, a scripted stand-in in the tests. Tokens substituted:

    {wall}      the MCP wall command, as one shell-quoted string
    {mcp_config} the wall as an `{"mcpServers": ...}` JSON string, the form Claude Code,
                Cursor and Claude Desktop take (`claude -p --mcp-config '{mcp_config}'`)
    {wall_url}  base URL of an HTTP wall started for this trial    (see http_wall)
    {task}      the task prompt from the scenario
    {sandbox}   a fresh directory for this trial

Three transports reach the same wall and write the same record: MCP over stdio
(`{wall}`), HTTP (`{wall_url}`), and in-process (`--in-process`: no token; the agent
builds `ajantik.adapter.Wall.from_env()` from variables set for it).

Write the agent command with absolute paths: the trial's cwd is that fresh directory,
so a relative path resolves against it and not against the repository.

On disk a round is `round_dir/round.json` (the manifest), `records/` (one JSONL session
record per trial, plus the agent's stdout and stderr) and `sandbox/`. Readers follow each
trial's `record` path in the manifest, so older lab layouts (ajantik.legacy) read the same.

    python -m ajantik.rounds --scenario examples/intake-form/scenario.yaml --out round-01 --reps 1 \
        -- goose run --with-extension 'w:{wall}' -t '{task}'

Nothing here calls a model. Whatever the agent spends, it spends on its own account:
this process cannot see or cap it (`BudgetGuard` only covers the lab's own harness).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ajantik import legacy
from ajantik.adapter import ENV_FAULT, ENV_RECORD, ENV_SCENARIO
from ajantik.http_wall import READY_PREFIX
from ajantik.scenario import Scenario, load_scenario

WALL = "{wall}"
WALL_URL = "{wall_url}"
MCP_CONFIG = "{mcp_config}"
TASK = "{task}"
SANDBOX = "{sandbox}"

DEFAULT_TIMEOUT_S = 300
RECORDS_DIR = "records"
# The manifest keeps its original file name: verdicts, page and labelling read it by
# this name from every recorded round.
MANIFEST = "round.json"


def manifest_path(round_dir: Path) -> Path:
    """The round's manifest, under its current name or the name older rounds used."""
    current = round_dir / MANIFEST
    if current.exists():
        return current
    older = [round_dir / name for name in legacy.MANIFESTS if (round_dir / name).exists()]
    return older[0] if older else current


@dataclass
class Trial:
    fault: str
    rep: int
    record: str
    sandbox: str
    command: list[str]
    exit_code: int | None
    duration_s: float
    timed_out: bool
    started_at: str


def wall_command(scenario_path: Path, fault_id: str, record_path: Path,
                 python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "ajantik.wall",
            "--scenario", str(scenario_path), "--fault", fault_id, "--record", str(record_path)]


def http_wall_command(scenario_path: Path, fault_id: str, record_path: Path,
                      python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "ajantik.http_wall",
            "--scenario", str(scenario_path), "--fault", fault_id, "--record", str(record_path)]


def transport(template: list[str], in_process: bool = False) -> str:
    if in_process:
        return "in-process"
    if any(WALL_URL in part for part in template):
        return "http"
    return "mcp"


def mcp_config(wall: list[str], name: str = "ajantik") -> str:
    return json.dumps({"mcpServers": {name: {"command": wall[0], "args": wall[1:]}}})


def _substitute(template: list[str], wall: str, task: str, sandbox: str,
                wall_url: str = "") -> list[str]:
    # {wall_url} first: "{wall}" is a prefix of it.
    config = mcp_config(shlex.split(wall)) if wall else ""
    return [part.replace(WALL_URL, wall_url).replace(MCP_CONFIG, config)
            .replace(WALL, wall).replace(TASK, task).replace(SANDBOX, sandbox)
            for part in template]


def _start_http_wall(command: list[str]) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline() if proc.stdout else ""
    if not line.startswith(READY_PREFIX):
        proc.kill()
        err = proc.stderr.read() if proc.stderr else ""
        raise SystemExit(f"The HTTP wall did not start: {line or err}")
    return proc, line[len(READY_PREFIX):].strip()


def _stop_http_wall(proc: subprocess.Popen) -> None:
    """SIGTERM makes the wall write session_end; a kill would lose the final state."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def task_prompt(scenario: Scenario) -> str:
    if not scenario.tasks:
        raise SystemExit("The scenario has no task.")
    if len(scenario.tasks) > 1:
        raise SystemExit(f"This runner expects a scenario with one task; it has "
                         f"{len(scenario.tasks)}. Cut the scenario down to one task or run "
                         "separate rounds.")
    return scenario.tasks[0].prompt


def fault_ids(scenario: Scenario, only: list[str] | None) -> list[str]:
    from ajantik.faults import canonical_id

    known = [f.id for f in scenario.faults]
    if not only:
        return known
    only = [canonical_id(f) for f in only]
    unknown = [f for f in only if f not in known]
    if unknown:
        raise SystemExit(f"Not in the scenario: {', '.join(unknown)}. It has: {', '.join(known)}")
    return only


def run_round(scenario_path: Path, out_dir: Path, agent_template: list[str], reps: int = 1,
              only: list[str] | None = None, timeout_s: int = DEFAULT_TIMEOUT_S,
              before_each: list[str] | None = None, dry_run: bool = False,
              python: str | None = None, in_process: bool = False) -> dict:
    # Absolute: every trial runs in a fresh directory, so a relative path in the wall
    # command (or in the agent command the caller writes) resolves against the wrong cwd.
    scenario_path = scenario_path.resolve()
    out_dir = out_dir.resolve()
    scenario = load_scenario(scenario_path)
    task = task_prompt(scenario)
    faults = fault_ids(scenario, only)
    mode = transport(agent_template, in_process)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = out_dir / RECORDS_DIR
    records.mkdir(exist_ok=True)

    trials: list[Trial] = []
    for fault in faults:
        slug = fault.replace(":", "_").replace("/", "_")
        for rep in range(1, reps + 1):
            stem = f"{slug}-{rep:03d}"
            record = records / f"{stem}.jsonl"
            # Fresh per trial: the wall is a new process, and the agent gets an empty
            # directory. Reusing either lets one trial's leftovers decide the next.
            sandbox = out_dir / "sandbox" / stem
            if sandbox.exists():
                shutil.rmtree(sandbox)
            sandbox.mkdir(parents=True)

            wall = shlex.join(wall_command(scenario_path, fault, record, python))
            wall_url = "http://wall.invalid" if dry_run and mode == "http" else ""
            command = _substitute(agent_template, wall, task, str(sandbox), wall_url)
            trial = Trial(fault=fault, rep=rep, record=str(record.relative_to(out_dir)),
                          sandbox=str(sandbox.relative_to(out_dir)), command=command,
                          exit_code=None, duration_s=0.0, timed_out=False,
                          started_at=datetime.now(UTC).isoformat(timespec="seconds"))
            if dry_run:
                trials.append(trial)
                print(shlex.join(command))
                continue

            if before_each:
                subprocess.run(_substitute(before_each, wall, task, str(sandbox)),
                               check=False, capture_output=True)
            env = None
            http_proc = None
            if mode == "in-process":
                env = {**os.environ, ENV_SCENARIO: str(scenario_path), ENV_FAULT: fault,
                       ENV_RECORD: str(record)}
            elif mode == "http":
                http_proc, wall_url = _start_http_wall(
                    http_wall_command(scenario_path, fault, record, python))
                command = _substitute(agent_template, wall, task, str(sandbox), wall_url)
                trial.command = command
            t0 = time.monotonic()
            try:
                done = subprocess.run(command, cwd=sandbox, capture_output=True, text=True,
                                      timeout=timeout_s, check=False, env=env)
            except FileNotFoundError as exc:
                # The commonest mistake, and it would read as an agent failure if swallowed:
                # a relative path in the agent command, resolved against the fresh cwd.
                raise SystemExit(
                    f"Agent command not found: {exc.filename}\n"
                    f"Every trial runs in a fresh directory ({sandbox}), so relative paths "
                    f"resolve against that directory, not the repository. Write the agent "
                    f"command with absolute paths."
                ) from exc
            except subprocess.TimeoutExpired:
                # The judge will find no session_end and report inconclusive. A timeout
                # must never be scored as behaviour.
                trial.timed_out = True
            else:
                trial.exit_code = done.returncode
                for stream, text in (("stdout", done.stdout), ("stderr", done.stderr)):
                    (records / f"{stem}.agent-{stream}.txt").write_text(text or "", encoding="utf-8")
            finally:
                if http_proc is not None:
                    _stop_http_wall(http_proc)
            trial.duration_s = round(time.monotonic() - t0, 2)
            trials.append(trial)
            print(f"{stem}: exit={trial.exit_code} {trial.duration_s}s"
                  f"{' TIMEOUT' if trial.timed_out else ''}", file=sys.stderr)

    manifest = {
        "format": "ajantik.round.v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario": str(scenario_path),
        "task": task,
        "agent_template": agent_template,
        "transport": mode,
        "before_each": before_each,
        "timeout_s": timeout_s,
        "reps": reps,
        "faults": faults,
        "trials": [asdict(t) for t in trials],
        "limitations": [
            ("This process never calls a model. Model, provider and spend belong to the agent "
             "and are not observable here."),
            ("The wall and the agent's working directory are fresh per trial. State the agent "
             "keeps elsewhere (its own session or memory store) is NOT isolated by this runner; "
             "use --before-each to clear it."),
            ("A timed-out trial has no session_end, so the judge records it as inconclusive, "
             "never as behaviour."),
        ],
    }
    if not dry_run:
        (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ajantik round", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, type=Path, dest="scenario")
    p.add_argument("--out", required=True, type=Path, dest="out",
                   help="directory the round is written to")
    p.add_argument("--reps", type=int, default=1, dest="reps",
                   help="repetitions per fault (default 1)")
    p.add_argument("--fault", action="append", dest="faults",
                   help="only this fault (repeatable); default: every fault in the scenario")
    p.add_argument("--timeout", "--zaman-asimi", type=int, default=DEFAULT_TIMEOUT_S,
                   dest="timeout", help="seconds per trial")
    p.add_argument("--before-each", dest="before_each",
                   help="command run before every trial (shell syntax)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the commands, run nothing")
    p.add_argument("--in-process", action="store_true",
                   help="the agent builds ajantik.adapter.Wall.from_env() itself; no wall token")
    # Everything after `--` is the agent command. A separator, not repeated --agent-arg
    # flags: agent commands are made of --flags, and argparse would claim them as its own.
    p.add_argument("agent", nargs=argparse.REMAINDER,
                   help=f"-- <agent command>; {WALL}, {WALL_URL}, {TASK} and {SANDBOX} "
                        "are substituted")
    args = p.parse_args(argv)

    template = args.agent
    if template and template[0] == "--":
        template = template[1:]
    if not template:
        raise SystemExit("STOP: no agent command. After the options write `-- <agent command>`, "
                         f"e.g. -- goose run --with-extension 'w:{WALL}' -t '{TASK}'")
    joined = " ".join(template)
    if not args.in_process and not any(tok in joined
                                       for tok in (WALL, WALL_URL, MCP_CONFIG)):
        raise SystemExit(f"STOP: the agent command has no {WALL} or {WALL_URL}, so the agent "
                         "never reaches the wall and the round runs without a fault. Pass the "
                         "wall as the agent's MCP server or base URL, or use --in-process.")
    if TASK not in joined:
        raise SystemExit(f"STOP: the agent command has no {TASK}, so the agent gets no task.")

    run_round(args.scenario, args.out, template, reps=args.reps, only=args.faults,
              timeout_s=args.timeout,
              before_each=shlex.split(args.before_each) if args.before_each else None,
              dry_run=args.dry_run, in_process=args.in_process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
