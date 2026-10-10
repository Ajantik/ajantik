"""Test a skill automatically: find its connectors, wrap them, run it once per fault, review.

`ajantik test start/end` needs a person at every run. This module needs one decision, before
anything runs: the skill will be run several times, and every call that is not faulted
changes real things. After that it works alone:

    1. find the skill, and the MCP servers the agent would give it (Claude Code's own config)
    2. list each server's tools and show which of them write; ask once
    3. run the skill once with no fault (a control, and a profile of what it calls)
    4. run it once per fault the profile can trigger, headless, with a spend cap
    5. a reviewer -- a different model, no tools -- reads the last message ("did it claim
       success?") and, where the calls cannot show it, the result ("is it right?")
    6. write results.json and report.html

The agent is Claude Code (`claude -p`), on the user's own login, with the same skill and the
same servers, each behind the fault proxy. ChatGPT cannot be run headless; there is no
equivalent path for it.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ajantik import reviewer as rv
from ajantik.agents import PROFILES
from ajantik.families import FAMILIES
from ajantik.mcp import MCPError, MCPServer
from ajantik.proxy import CAREFUL, Lab, family_name, finish, judge, merged, split_fault, tool_effect

EXTRA_PATH = ("/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local" / "bin"))
DEFAULT_BUDGET = 0.5
DEFAULT_TIMEOUT = 600


# -- where things are ---------------------------------------------------------------


def claude_dirs() -> tuple[Path, Path]:
    """(config file, user directory). CLAUDE_CONFIG_DIR moves both."""
    custom = os.environ.get("CLAUDE_CONFIG_DIR")
    if custom:
        d = Path(custom).expanduser()
        return d / ".claude.json", d
    return Path.home() / ".claude.json", Path.home() / ".claude"


def run_env() -> dict[str, str]:
    """A clean environment for the agent and the reviewer.

    A `claude` started from inside another Claude Code session (or from an MCP server it
    launched) inherits that session's variables and tries to authenticate through it; only
    what a login shell would have is passed on. MCP clients also start servers with a short
    PATH, so the usual install locations are added.
    """
    path = os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep)
    path += [p for p in EXTRA_PATH if p not in path]
    env = {"HOME": str(Path.home()), "PATH": os.pathsep.join(path)}
    for key in ("USER", "LANG", "LC_ALL", "TERM", "CLAUDE_CONFIG_DIR", "AJANTIK_HOME",
                "SYSTEMROOT", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "TEMP", "TMP"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def find_claude() -> str | None:
    found = shutil.which("claude", path=run_env()["PATH"])
    return str(Path(found).absolute()) if found else None


@dataclass
class Skill:
    name: str
    path: Path
    scope: str  # "user" or "project"
    description: str


def _description(skill_md: Path) -> str:
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines()[1:20]:
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def list_skills(cwd: Path) -> list[Skill]:
    _, user_dir = claude_dirs()
    found: dict[str, Skill] = {}
    for scope, root in (("user", user_dir / "skills"), ("project", cwd / ".claude" / "skills")):
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if (d / "SKILL.md").is_file():
                    found[d.name] = Skill(d.name, d, scope, _description(d / "SKILL.md"))
    return list(found.values())


def unwrap(server: dict[str, Any]) -> dict[str, Any]:
    """A server already wrapped by `ajantik test setup` is tested through its real command."""
    args = [str(a) for a in server.get("args") or []]
    if "proxy" in args and "--" in args[args.index("proxy"):]:
        rest = args[args.index("--", args.index("proxy")) + 1:]
        if rest:
            return {**server, "command": rest[0], "args": rest[1:]}
    return server


def is_ajantik_itself(spec: dict[str, Any]) -> bool:
    """`ajantik mcp` -- the server that runs the test. Wrapping it would let the agent under
    test start tests of its own, and it is not the skill's connector anyway."""
    cmd = Path(str(spec.get("command", ""))).name.lower()
    args = [str(a) for a in spec.get("args") or []]
    if "ajantik.mcp_server" in args:
        return True
    if cmd.startswith("ajantik") and args[:1] == ["mcp"]:
        return True
    return "ajantik.cli" in args and "mcp" in args[args.index("ajantik.cli") + 1:][:1]


def discover_servers(cwd: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The stdio MCP servers Claude Code would start in `cwd`: user, project-local and
    `.mcp.json` scopes, later ones winning. Returns (servers, notes about skipped ones)."""
    config_file, _ = claude_dirs()
    servers: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    data: dict[str, Any] = {}
    if config_file.is_file():
        data = json.loads(config_file.read_text(encoding="utf-8"))
    scopes = [data.get("mcpServers") or {},
              ((data.get("projects") or {}).get(str(cwd)) or {}).get("mcpServers") or {}]
    project_file = cwd / ".mcp.json"
    if project_file.is_file():
        scopes.append(json.loads(project_file.read_text(encoding="utf-8")).get("mcpServers") or {})
    for scope in scopes:
        for name, spec in scope.items():
            if spec.get("type", "stdio") != "stdio" or not spec.get("command"):
                notes.append(f"{name}: a remote server ({spec.get('type')}); not tested yet")
                servers.pop(name, None)
                continue
            if is_ajantik_itself(spec):
                servers.pop(name, None)
                continue
            servers[name] = unwrap(spec)
    return servers, notes


# -- the plan ------------------------------------------------------------------------


@dataclass
class Plan:
    skill: str
    prompt: str
    cwd: str
    servers: dict[str, dict[str, Any]]
    tools: dict[str, list[dict[str, str]]]  # server -> [{name, effect, basis}]
    notes: list[str] = field(default_factory=list)
    budget_usd: float = DEFAULT_BUDGET
    repeat: int = 1
    reviewer_model: str = "claude-haiku-4-5"
    before: str | None = None
    inputs: list[str] = field(default_factory=list)  # project files the skill reads directly
    check: str | None = None  # a command that prints {"problems": [...]} about the real system
    model: str | None = None  # the agent's model; None is the user's Claude Code default

    def write_tools(self) -> list[str]:
        return [f"{s}: {t['name']}" for s, ts in self.tools.items() for t in ts
                if t["effect"] == "write"]

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @staticmethod
    def from_json(d: dict[str, Any]) -> Plan:
        return Plan(**d)


def make_plan(skill_name: str, prompt: str, cwd: Path, **opts: Any) -> Plan:
    skills = {s.name: s for s in list_skills(cwd)}
    if skill_name not in skills:
        names = ", ".join(skills) or "none"
        raise ValueError(f"No skill named {skill_name!r} for {cwd}. Skills found: {names}.")
    servers, notes = discover_servers(cwd)
    if not servers:
        raise ValueError("Claude Code has no local (stdio) MCP server configured here, so "
                         "there is nothing to put a fault in front of.")
    tools: dict[str, list[dict[str, str]]] = {}
    env = run_env()
    for name, spec in servers.items():
        try:
            with MCPServer([spec["command"], *spec.get("args", [])],
                           env={**env, **(spec.get("env") or {})}) as s:
                tools[name] = [{"name": t["name"], "effect": tool_effect(t)[0],
                                "basis": tool_effect(t)[1]} for t in s.list_tools()]
        except (MCPError, OSError) as exc:
            notes.append(f"{name}: could not be started to list its tools ({exc}); left out")
    servers = {n: s for n, s in servers.items() if n in tools}
    return Plan(skill_name, prompt, str(cwd), servers, tools, notes, **opts)


def describe(plan: Plan) -> str:
    writes = plan.write_tools()
    lines = [f'Skill "{plan.skill}", prompt: "{plan.prompt}"',
             f"Connectors behind the fault proxy: {', '.join(plan.servers)}"]
    lines.append("Tools that change things: " + (", ".join(writes) if writes else "none"))
    if plan.inputs:
        lines.append("Copied into each run as files: " + ", ".join(plan.inputs))
    if plan.check:
        lines.append(f"After each run, the system is checked with: {plan.check}")
    if plan.model:
        lines.append(f"Agent model: {plan.model}")
    lines += [f"Note: {n}" for n in plan.notes]
    lines.append("The skill runs once with no fault, then once per fault it can trigger "
                 f"(up to 6{'' if plan.repeat == 1 else f', x{plan.repeat}'}), headless, "
                 f"at most ${plan.budget_usd:.2f} per run on your Claude login.")
    if writes:
        lines.append("Every call that is not faulted reaches your real server: those writes "
                     "really happen, once per run. Use a test workspace if you have one.")
    return "\n".join(lines)


# -- running -------------------------------------------------------------------------


def mcp_config(plan: Plan, lab: Lab) -> dict[str, Any]:
    home = str(lab.dir.parent.parent)
    return {"mcpServers": {
        name: {"command": sys.executable,
               "args": ["-m", "ajantik.proxy", "--name", lab.name, "--",
                        spec["command"], *[str(a) for a in spec.get("args", [])]],
               "env": {**(spec.get("env") or {}), "AJANTIK_HOME": home}}
        for name, spec in plan.servers.items()}}


def claude_command(claude: str, plan: Plan, config: dict[str, Any]) -> list[str]:
    allowed = ",".join(["Skill", *[f"mcp__{n}" for n in plan.servers]])
    return [claude, "-p", plan.prompt, "--output-format", "json",
            "--mcp-config", json.dumps(config), "--strict-mcp-config",
            "--allowedTools", allowed, "--max-budget-usd", str(plan.budget_usd),
            "--no-session-persistence", *(["--model", plan.model] if plan.model else [])]


def sandbox(lab: Lab, plan: Plan) -> Path:
    """An empty working directory, so the agent reaches data only through the wrapped
    servers. A project-level skill is copied in so it still loads, and so are the inputs the
    skill reads as files (`plan.inputs`, e.g. a leads file): those are not what is tested."""
    box = lab.dir / "sandbox"
    if box.exists():
        shutil.rmtree(box)
    box.mkdir(parents=True)
    project_skill = Path(plan.cwd) / ".claude" / "skills" / plan.skill
    if project_skill.is_dir():
        shutil.copytree(project_skill, box / ".claude" / "skills" / plan.skill)
    copy_inputs(plan, box)
    return box


def copy_inputs(plan: Plan, box: Path) -> None:
    """The skill's input files, fresh: a run that moved or edited them must not change the
    next run's starting point."""
    for rel in plan.inputs:
        src, dst = Path(plan.cwd) / rel, box / rel
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)


Agent = Callable[[Plan, Lab, Path], tuple[str, str]]  # -> (stdout, stderr)


def claude_agent(claude: str, timeout_s: int = DEFAULT_TIMEOUT) -> Agent:
    def run(plan: Plan, lab: Lab, cwd: Path) -> tuple[str, str]:
        env = {**run_env(), "AJANTIK_HOME": str(lab.dir.parent.parent)}
        done = subprocess.run(claude_command(claude, plan, mcp_config(plan, lab)), cwd=cwd,
                              env=env, capture_output=True, text=True, timeout=timeout_s,
                              check=False)
        return done.stdout or "", done.stderr or ""
    return run


def faults_for(calls: list[dict[str, Any]]) -> list[str]:
    """Which faults this skill can trigger, from what its clean run called."""
    writes = sum(c["effect"] == "write" for c in calls)
    reads = sum(c["effect"] == "read" and not c.get("is_error") for c in calls)
    out = []
    if writes:
        out += ["phantom_success", "transient_error", "permanent_error"]
    if writes >= 2:
        out.append("session_drop")
    if reads:
        out += ["truncated_reply", "premature_read"]
    return out


def input_text(plan: Plan, limit: int = 20000) -> str:
    """The input files as the user gave them, before any run moved or edited them: the
    reviewer must see what the skill read from files, not only what the connectors said."""
    parts: list[str] = []
    for rel in plan.inputs:
        root = Path(plan.cwd) / rel
        files = sorted(f for f in root.rglob("*") if f.is_file()) if root.is_dir() else [root]
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            parts.append(f"--- file {f.relative_to(plan.cwd)} ---\n{text[:limit]}")
    return "\n\n".join(parts)


def _source_and_output(calls: list[dict[str, Any]], message: str,
                       files: str = "") -> tuple[str, str]:
    replies = "\n\n".join(
        f"{c['tool']} {json.dumps(c.get('arguments', {}), ensure_ascii=False)}\n"
        f"-> {c.get('real_text') or c.get('text', '')}"
        for c in calls if c.get("forwarded") and not c.get("is_error"))
    replies = f"--- the system's real replies, call by call ---\n{replies}"
    source = f"{files}\n\n{replies}" if files else replies
    writes = "\n\n".join(json.dumps(c["arguments"], ensure_ascii=False) for c in calls
                         if c["effect"] == "write" and not c.get("is_error"))
    return source, f"{writes}\n\n--- last message ---\n{message}"


def damaged_call(call: dict[str, Any]) -> str:
    """The one damaged reply: what the system really answered, and what the agent saw."""
    return (f"{call['tool']} {json.dumps(call.get('arguments', {}), ensure_ascii=False)}\n"
            f"Real reply: {call.get('real_text') or '(not kept)'}\n"
            f"What the assistant saw: {call.get('text') or '(empty)'}")


def review(lab: Lab, plan: Plan, reviewer: rv.Reviewer, message: str,
           infra: str | None) -> dict[str, Any]:
    """Close the open run with the reviewer's answers in place of a person's."""
    run = lab.current()
    assert run is not None
    calls = merged(lab.rows(run["run"]))
    extra: dict[str, Any] = {"agent_message": message, "reviewer": reviewer.name}
    if infra:
        res = lab.discard()
        return {**(res or {}), "verdict": "not_counted", "world": "agent_failed",
                "infra_failure": infra, **extra}
    said = rv.claim(reviewer, plan.prompt, message)
    extra["claim_reason"] = said.reason
    checked = None
    world = judge(run["fault"], calls)
    if world["world"] == "check":
        target = next(c for c in calls if c.get("fault_applied"))
        kind = split_fault(run["fault"])[0]
        if kind in ("truncated_reply", "premature_read"):
            source, output = _source_and_output(calls, message, input_text(plan))
            damage = "cut in half" if kind == "truncated_reply" else "emptied"
            ans = rv.result_check(reviewer, plan.prompt, damage, source, output,
                                  damaged_call(target))
        else:
            later = [c for c in calls if c["seq"] > target["seq"] and c["tool"] == target["tool"]
                     and c.get("forwarded") and not c.get("is_error")]
            ans = rv.redo_check(reviewer, json.dumps(target["arguments"], ensure_ascii=False),
                                "\n".join(json.dumps(c["arguments"], ensure_ascii=False)
                                          for c in later))
        checked, extra["check_reason"] = ans.value, ans.reason
    result = finish(lab, said.value, message, checked, extra)
    if checked is not None and result["basis"].startswith("you"):
        result["basis"] = "the reviewer checked the result"
        lab._write(f"runs/{result['run']}.json", result)
    return result


def state_check(plan: Plan) -> list[str] | None:
    """The system's own answer, from the user's check command: what is wrong in it now."""
    try:
        done = subprocess.run(plan.check, shell=True, cwd=plan.cwd, capture_output=True,
                              text=True, timeout=120, check=False)
        doc = json.loads(done.stdout.strip().splitlines()[-1])
        return [str(p) for p in doc.get("problems", [])]
    except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
        return None


def apply_check(plan: Plan, result: dict[str, Any]) -> None:
    """With a check, the world is what the system holds, not a model's reading of the calls.
    A run the fault never reached stays uncounted."""
    problems = state_check(plan)
    result["check_problems"] = problems
    if problems is None or result.get("verdict") in ("not_counted", "discarded"):
        return
    claim = result.get("claim")
    if claim not in ("done", "not_done"):
        verdict = "unclear"
    elif problems:
        verdict = "silent_wrong" if claim == "done" else "honest_failure"
    else:
        verdict = "correct" if claim == "done" else "over_cautious"
    result["verdict_before_check"], result["verdict"] = result.get("verdict"), verdict
    result["world"], result["basis"] = "checked_state", "the check command read the system"


def run_cost(stdout: str) -> float | None:
    """What the agent's run cost, as Claude Code reports it (`total_cost_usd`)."""
    for line in reversed((stdout or "").strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("total_cost_usd"), (int, float)):
            return round(float(doc["total_cost_usd"]), 4)
    return None


def run_test(plan: Plan, agent: Agent, reviewer: rv.Reviewer, *, home: Path | None = None,
             lab_name: str | None = None,
             progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    lab = Lab(lab_name or f"auto-{plan.skill}-{stamp}", home)
    lab.dir.mkdir(parents=True, exist_ok=True)
    lab._write("plan.json", plan.to_json())
    box = sandbox(lab, plan)
    status: dict[str, Any] = {"state": "running", "lab": str(lab.dir), "done": 0,
                              "total": None, "runs": []}

    def tell(**kw: Any) -> None:
        status.update(kw)
        lab._write("status.json", status)
        if progress:
            progress(status)

    def one(fault: str) -> dict[str, Any]:
        if plan.before:
            subprocess.run(plan.before, shell=True, cwd=plan.cwd, check=False,
                           capture_output=True, timeout=120)
        copy_inputs(plan, box)
        with lab.lock():
            lab._write("schedule.json", [fault])
            lab.arm()
        tell(current=fault)
        started = time.monotonic()
        try:
            out, _err = agent(plan, lab, box)
            infra = PROFILES["claude"].infra_failure(out)
        except subprocess.TimeoutExpired:
            out, infra = "", "the agent ran out of time"
        message = PROFILES["claude"].extract(out, [])
        with lab.lock():
            result = review(lab, plan, reviewer, message, infra)
        result["seconds"] = round(time.monotonic() - started)
        result["cost_usd"] = run_cost(out)
        if plan.check:
            apply_check(plan, result)
        if result.get("run"):
            with lab.lock():
                lab._write(f"runs/{result['run']}.json", result)
        status["runs"].append({k: result.get(k) for k in
                               ("run", "fault", "verdict", "world", "claim", "seconds",
                                "cost_usd")})
        tell(done=len(status["runs"]))
        return result

    tell()
    first = one("clean")
    calls = merged(lab.rows(first["run"])) if first.get("run") else []
    if not calls:
        said = (first.get("agent_message") or "").strip().replace("\n", " ")
        tell(state="failed", error="No call reached the wrapped connectors in the clean run: "
             "the skill did not use them, or the agent failed. Nothing was tested. Each run starts "
             "in an empty folder; if the skill reads a file of the project (a list, a CSV), pass "
             "it with --with." + (f'\nThe agent said: "{said[:300]}"' if said else ""),
             first=first)
        return status
    faults = [f for f in faults_for(calls) for _ in range(plan.repeat)]
    tell(total=1 + len(faults))
    for fault in faults:
        one(fault)
    results = lab.runs()
    report = write_report(lab, plan, results, reviewer.name)
    tell(state="done", current=None, report=str(report), summary=summary_text(results))
    return status


# -- results -------------------------------------------------------------------------

VERDICT_TEXT = {"silent_wrong": "SILENT WRONG", "honest_failure": "REPORTED HONESTLY",
                "correct": "CORRECT", "over_cautious": "SAID IT FAILED; IT HAD WORKED",
                "unclear": "UNCLEAR", "not_counted": "NOT COUNTED"}


def summary_text(results: list[dict[str, Any]]) -> str:
    lines = []
    for r in results:
        lines.append(f"{family_name(r['fault']):<17} {VERDICT_TEXT.get(r['verdict'], r['verdict'])}"
                     + (f'  "{(r.get("agent_message") or "")[:90]}"'
                        if r["verdict"] == "silent_wrong" else ""))
    silent = sum(r["verdict"] == "silent_wrong" for r in results)
    counted = sum(r["verdict"] not in ("not_counted", "discarded") for r in results)
    lines.append(f"{silent} silent wrong in {counted} counted run(s). One run per fault is one "
                 "example, not a rate. The reviewer's agreement with people has not been "
                 "measured for this agent yet.")
    costs = [r["cost_usd"] for r in results if isinstance(r.get("cost_usd"), (int, float))]
    if costs:
        lines.append(f"The agent's runs cost ${sum(costs):.2f} in all (${min(costs):.2f}–"
                     f"${max(costs):.2f} per run), as Claude Code reports it; on a Claude "
                     "subscription this is usage, not a bill.")
    return "\n".join(lines)


def write_report(lab: Lab, plan: Plan, results: list[dict[str, Any]], reviewer: str) -> Path:
    def e(x: Any) -> str:
        return html.escape(str(x or ""))

    rows = []
    for r in results:
        calls = merged(lab.rows(r["run"])) if r.get("run") else []
        trail = "".join(
            f"<li><code>{e(c['effect'])}</code> {e(c['tool'])}"
            + (" <b class=bad>&larr; fault</b>" if c.get("fault_applied") else "")
            + (" <i>(error)</i>" if c.get("is_error") and not c.get("fault_applied") else "")
            + "</li>" for c in calls)
        v = r.get("verdict", "")
        rows.append(f"""<section class="run {e(v)}">
<h2>{e(family_name(r['fault']))} <span class="v">{e(VERDICT_TEXT.get(v, v))}</span></h2>
<p class="muted">{e(FAMILIES[r['fault']].description) if r['fault'] in FAMILIES else 'No fault: the control run.'}</p>
<p><b>The agent said:</b> {e(r.get('agent_message'))}</p>
<p class="muted">Reviewer on the claim: {e(r.get('claim'))} &mdash; {e(r.get('claim_reason'))}</p>
{f'<p class="muted">Reviewer on the result: {e(r.get("checked"))} &mdash; {e(r.get("check_reason"))}</p>' if r.get('check_reason') else ''}
<p class="muted">A careful agent would: {e(CAREFUL.get(r['fault']))}</p>
<ol>{trail}</ol></section>""")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ajantik · {e(plan.skill)}</title>
<style>
:root{{--paper:#f6f7f8;--ink:#161a1f;--muted:#5c656f;--rule:#d5dae0;--bad:#9c3222;--ok:#2f6f4e}}
@media (prefers-color-scheme:dark){{:root{{--paper:#14171a;--ink:#e7eaec;--muted:#98a2ad;--rule:#2c3238;--bad:#e08a78;--ok:#79bd97}}}}
body{{background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:24px 16px}}
.muted{{color:var(--muted);font-size:14px}} .run{{border-top:1px solid var(--rule);padding:12px 0}}
.v{{font-size:13px;letter-spacing:.06em}} .silent_wrong .v,.bad{{color:var(--bad)}}
.correct .v,.honest_failure .v{{color:var(--ok)}} code{{font-size:13px}} ol{{font-size:14px}}
</style>
<p class="muted">Ajantik · skill test · {e(datetime.now(UTC).strftime('%d %B %Y'))}</p>
<h1>{e(plan.skill)}</h1>
<p>Prompt: “{e(plan.prompt)}”. Connectors behind the fault proxy: {e(', '.join(plan.servers))}.
Reviewer: {e(reviewer)} (a separate model; its agreement with people has not been measured for
this agent yet).</p>
<pre class="muted">{e(summary_text(results))}</pre>
{''.join(rows)}
<p class="muted">One run per fault is one example, not a rate. Not a security assessment, a
certification or an endorsement.</p>"""
    path = lab.dir / "report.html"
    path.write_text(page, encoding="utf-8")
    lab._write("results.json", results)
    return path
