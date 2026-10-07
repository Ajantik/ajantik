"""Command line.

The end-to-end path for testing an agent you do not own:

    ajantik record   one real run through a recording proxy -> scenario.yaml
    ajantik round    every fault x N repetitions against the wall (MCP, HTTP or in-process)
    ajantik verdicts world state x agent claim -> verdicts.json
    ajantik report   verdicts -> a self-contained HTML report
    ajantik judge-sample / judge-agreement   measure the judge on a new agent

"""

from __future__ import annotations

import json
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path

import typer

from ajantik import HARNESS
from ajantik.identity import Identity
from ajantik.pricing import PRICES, worst_case_call_usd
from ajantik.report import badge_svg, build_comparison, build_report, load_trials
from ajantik.scenario import load_scenario
from ajantik.skill import load_skill
from ajantik.trial import BudgetGuard, run_trial

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Ajantik: test how agents behave when their tools lie or fail.")
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def command(name: str, **kw):
    """Register a command under its name."""
    def wrap(fn):
        app.command(name, **kw)(fn)
        return fn
    return wrap


def _rounds(specs: list[str]) -> dict[str, Path]:
    from ajantik.verdicts import parse_rounds

    return parse_rounds(specs)


# -- the end-to-end path -------------------------------------------------------------


@command("record", context_settings=PASSTHROUGH)
def record_cmd(
    ctx: typer.Context,
    server: str = typer.Option(..., help="Your real MCP server command, as one shell string."),
    task: str = typer.Option(..., help="The task given to the agent, one sentence."),
    out: Path = typer.Option(..., help="Scenario file to write (.yaml)."),
    work: Path | None = typer.Option(None, help="Where the reference run is kept "
                                                "(default: next to --out)."),
    timeout: int = typer.Option(300, help="Seconds for the reference run."),
    allow_real_calls: bool = typer.Option(
        False, "--allow-real-calls",
        help="Required: the reference run calls your REAL server, with real side effects."),
) -> None:
    """Run the agent once through a recording proxy and write the scenario from that run.

    Everything after `--` is the agent command; {wall} (or {mcp_config}) and {task} are
    substituted, as in `ajantik round`.
    """
    from ajantik.record import build_scenario, record, render

    agent = [a for a in ctx.args if a != "--"]
    if not agent:
        raise typer.BadParameter("No agent command. Add `-- <agent command with {wall}>`.")
    if not allow_real_calls:
        typer.echo("STOP: the reference run calls your real server with real side effects. "
                   "Point it at a test account or sandbox, then pass --allow-real-calls.")
        raise typer.Exit(2)
    if out.exists():
        raise typer.BadParameter(f"{out} exists; not overwriting it.")
    server_cmd = shlex.split(server)
    log, _ = record(server_cmd, task, agent, work or out.parent / f"{out.stem}-reference",
                    timeout_s=timeout)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    data, notes = build_scenario(rows, task, server_cmd)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, notes), encoding="utf-8")
    checks = data["tasks"][0]["state_checks"]
    typer.echo(f"{len(data['tools'])} tools, {data['recorded_from']['calls']} calls recorded "
               f"-> {out}")
    typer.echo("Inferred (review these before spending anything):")
    for n in notes:
        typer.echo(f"  - {n}")
    typer.echo("State checks taken from the reference run (true only if that run was right):")
    for c in checks:
        typer.echo(f"  {c['key']} == {c['equals']!r}")
    typer.echo(f"\nNext: ajantik faults {out}   then   ajantik round --scenario {out} ...")


@command("round", context_settings=PASSTHROUGH, add_help_option=False)
def round_cmd(ctx: typer.Context) -> None:
    """Run every fault x N repetitions against an agent (see `ajantik round --help`)."""
    from ajantik.rounds import main

    raise typer.Exit(main(list(ctx.args)))


@command("verdicts", context_settings=PASSTHROUGH, add_help_option=False)
def verdicts_cmd(ctx: typer.Context) -> None:
    """World state x agent claim for a round (see `ajantik verdicts --help`)."""
    from ajantik.verdicts import main

    raise typer.Exit(main(list(ctx.args)))


@command("report")
def report_cmd(
    scenario: Path = typer.Option(..., help="The scenario the round ran."),
    round_: list[str] = typer.Option(..., "--round", help="label=directory (repeatable)."),
    out: Path = typer.Option(..., help="HTML file to write."),
    title: str = typer.Option("Crash Test Report", help="Page title."),
) -> None:
    """Write a self-contained HTML report: matrix, recorded sessions, judge, limits."""
    from ajantik.page import page_data, render

    data = page_data(scenario, _rounds(round_))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, title), encoding="utf-8")
    typer.echo(f"Report -> {out}")
    for j in data["judge"]:
        if not j["measured"]:
            typer.echo(f"WARNING: {j['warning']}")


@command("judge-sample")
def judge_sample(
    scenario: Path = typer.Option(..., help="The scenario the round ran."),
    round_: list[str] = typer.Option(..., "--round", help="label=directory (repeatable)."),
    out: Path = typer.Option(..., help="Directory for set.yaml, form.yaml and key.yaml."),
    size: int = typer.Option(20, help="Items to label (stratified sample)."),
) -> None:
    """Write a blind labelling form from a round's wrong-world cases."""
    from ajantik.judge import sample_form

    set_path, form, key, n, total = sample_form(scenario, _rounds(round_), out, size)
    typer.echo(f"{n} of {total} cases -> {form}")
    typer.echo(f"Do not open {set_path.name} or {key.name} until every item is labelled.")
    typer.echo(f"Then: ajantik judge-agreement --dir {out} --agent <name> --register")


@command("judge-agreement")
def judge_agreement(
    dir_: Path = typer.Option(..., "--dir", help="The directory judge-sample wrote."),
    agent: str = typer.Option(..., help="Agent the round ran, e.g. claude-code."),
    language: str = typer.Option("", help="tr or en (default: read from the texts)."),
    register: bool = typer.Option(False, help="Record the measurement in ajantik-judge.yaml."),
    source: str = typer.Option("", help="Where the measurement is documented."),
) -> None:
    """Kappa and the direction matrix between the filled form and the judge."""
    from ajantik import judge
    from ajantik.agreement import get, load_items
    from ajantik.oracle import CURRENT

    texts = " ".join(str(get(i, "text") or "") for i in load_items(dir_ / "set.yaml"))
    lang = language or judge.language_of(texts)
    m = judge.measure(dir_ / "set.yaml", dir_ / "form.yaml", dir_ / "key.yaml", agent, lang,
                      CURRENT, source or str(dir_), datetime.now(UTC).date().isoformat())
    typer.echo(f"{agent} ({lang}), judge v{m.oracle}: kappa {m.kappa} "
               f"({judge.interval_text(m)}), {m.agree}/{m.decisive} decisive "
               f"labels matched, {m.labelled} labelled")
    typer.echo("                 judge: silent   judge: honest")
    typer.echo(f"human: silent    {m.human_silent_judge_silent:>13}   "
               f"{m.human_silent_judge_honest:>13}")
    typer.echo(f"human: honest    {m.human_honest_judge_silent:>13}   "
               f"{m.human_honest_judge_honest:>13}")
    typer.echo("Publish kappa and the matrix together: kappa alone makes a cautious judge "
               "look worse than it is, and the matrix alone hides its abstentions.")
    if register:
        judge.register(m)
        typer.echo(f"Registered in {judge.LOCAL_FILE}")


@command("demo")
def demo_cmd(
    out: Path = typer.Option(Path("ajantik-demo"), help="Directory for the rounds and report."),
    scenario: Path | None = typer.Option(None, help="Scenario (default: the support ticket)."),
    reps: int = typer.Option(3, help="Repetitions per fault."),
) -> None:
    """Thirty seconds, no API key: a blind and a verifying scripted agent, one report."""
    from ajantik.demo import DEFAULT_SCENARIO, run

    scen = scenario or DEFAULT_SCENARIO
    if not scen.exists():
        raise typer.BadParameter(f"No scenario at {scen}; pass --scenario.")
    report = run(scen, out, reps)
    typer.echo(f"Report: {report}")
    typer.echo("The blind agent says 'Done' over a save that never happened; the verifying "
               "one reads it back and catches it. Open the report and click a cell.")


# -- the lab's own harness and tools ------------------------------------------------


@command("identity")
def identity_cmd(path: Path, model: str = "claude-opus-5", effort: str = "medium") -> None:
    """Show the identity of a skill folder or zip."""
    skill = load_skill(path)
    ident = Identity(skill.fingerprint, model, effort, HARNESS)
    typer.echo(f"{skill.name}: identity {ident.id} · recipe {skill.fingerprint[:12]} · "
               f"{len(skill.files)} files")


@command("skill-test")
def skill_test(
    scenario: Path,
    budget: float = typer.Option(1.50, "--budget", help="Hard spend cap (USD)."),
    reps: int = typer.Option(3, "--reps", help="Trials per task x condition."),
    model: str = "claude-opus-5",
    effort: str = "medium",
    max_tokens: int = 4000,
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Show the plan and worst-case cost; call nothing."),
    data: Path = typer.Option(Path("data"), "--data"),
    skill_path: Path | None = typer.Option(
        None, "--skill", help="Test this skill folder or zip instead of the scenario's."),
    conditions: str = typer.Option("", "--conditions",
                                   help="Only these conditions (comma-separated ids)."),
    extra_skill: list[Path] = typer.Option(
        [], "--extra-skill",
        help="Install another skill alongside (interaction test); repeatable."),
) -> None:
    """Run the scenario's skill under clean and faulty conditions; write its track record."""
    if model not in PRICES:
        raise typer.BadParameter(f"No price known for model: {model}")
    scen = load_scenario(scenario)
    from ajantik.skill import combine

    skill = combine(load_skill(skill_path or scen.skill_path),
                    [load_skill(p) for p in extra_skill])
    harness = HARNESS
    if scen.mcp_server:
        from ajantik.mcp import check_supported, harness_suffix

        check_supported(scen)  # refuses state_checks before anything is spent
        harness += harness_suffix(scen.mcp_server)
    ident = Identity(skill.fingerprint, model, effort, harness)
    if conditions:
        wanted = {k.strip() for k in conditions.split(",")}
        scen.faults = [f for f in scen.faults if f.id in wanted]
        if not scen.faults:
            raise typer.BadParameter(f"The scenario has none of these conditions: {conditions}")
    store = data / "trials" / f"{ident.id}.jsonl"
    done: dict[tuple[str, str], int] = {}
    evidence: dict[tuple[str, str], int] = {}
    for t in load_trials(store):  # a record accumulates: new reps continue the numbering
        key = (t["task"], t["fault"])
        done[key] = max(done.get(key, 0), t["rep"])
        evidence[key] = evidence.get(key, 0) + (t["success"] is not None)
    plan = [
        (t, f, done.get((t.id, f.id), 0) + r, evidence.get((t.id, f.id), 0) + r)
        for t in scen.tasks
        for f in scen.faults
        for r in range(1, reps + 1)
    ]
    # Least evidence first: a budget stop leaves every condition with as many finished trials
    # as possible, and a condition cut short earlier goes to the front, not the back.
    plan.sort(key=lambda p: (p[3], scen.tasks.index(p[0]), scen.faults.index(p[1])))
    worst_call = worst_case_call_usd(model, 6000, max_tokens)
    if scen.mcp_server:
        typer.echo(f"Bench: REAL MCP server — {' '.join(scen.mcp_server)}")
        typer.echo("No state oracle (the server's state is its own); the identity keeps a "
                   "separate record.")
    typer.echo(
        f"{' + '.join([skill.name, *(s.name for s in skill.co)])} · identity {ident.id} · "
        f"{len(plan)} trials ({len(scen.tasks)} tasks × {len(scen.faults)} conditions × "
        f"{reps}) · cap ${budget:.2f}"
    )
    typer.echo(f"Rough worst case per call ~${worst_call:.3f}; stops before the cap is crossed.")
    if dry_run:
        typer.echo("Order (least evidence first): "
                   + ", ".join(f"{f.id}#{r}" for _, f, r, _ in plan[:5])
                   + (" …" if len(plan) > 5 else ""))
        return

    import anthropic

    client = anthropic.Anthropic()
    guard = BudgetGuard(cap_usd=budget)
    store.parent.mkdir(parents=True, exist_ok=True)
    for task, fault, rep, _ in plan:
        if scen.mcp_server:
            # A fresh server per trial: a trial that inherited the previous one's state
            # would not be an independent sample.
            from ajantik.mcp import MCPServer, MCPTools

            with MCPServer(scen.mcp_server) as server:
                res = run_trial(client, skill, scen, task, fault, ident, rep, guard,
                                max_tokens=max_tokens, tools=MCPTools(server, fault))
        else:
            res = run_trial(client, skill, scen, task, fault, ident, rep, guard,
                            max_tokens=max_tokens)
        with store.open("a", encoding="utf-8") as fh:
            fh.write(res.to_json() + "\n")
        mark = {True: "✓", False: "✗", None: "·"}[res.success]
        typer.echo(
            f"{mark} {task.id}/{fault.id} #{rep}: {res.stop}, {res.turns} turns, "
            f"${res.cost_usd:.4f} (total ${guard.spent_usd:.4f})"
        )
        if res.stop == "budget":
            typer.echo(f"Stopped at the cap: {res.error}")
            break

    trials = load_trials(store)
    report, summary = build_report(ident, skill.name, trials)
    out = data / "records" / ident.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.md").write_text(report, encoding="utf-8")
    (out / "badge.svg").write_text(badge_svg(summary), encoding="utf-8")
    typer.echo(f"\nReport: {out / 'record.md'}\nBadge: {out / 'badge.svg'}")


@command("suggest")
def suggest_cmd(
    scenario: Path,
    identity_id: str = typer.Option(..., "--identity",
                                    help="Identity whose trials to examine."),
    skill_path: Path = typer.Option(..., "--skill", help="That identity's skill folder."),
    out: Path = typer.Option(..., "--out",
                             help="Folder for the suggested skill."),
    budget: float = typer.Option(0.15, "--budget",
                                 help="Hard cap for the suggestion call (USD)."),
    model: str = "claude-opus-5",
    data: Path = typer.Option(Path("data"), "--data"),
) -> None:
    """Suggest rules from failed trials and write a hardened SKILL.md."""
    import anthropic

    from ajantik.suggest import suggest

    scen = load_scenario(scenario)
    skill = load_skill(skill_path)
    trials = load_trials(data / "trials" / f"{identity_id}.jsonl")
    if not any(t["success"] is False for t in trials):
        raise typer.BadParameter("This identity has no failed trial; no rule to suggest.")
    suggestion, cost = suggest(anthropic.Anthropic(), skill, scen, trials, model,
                               BudgetGuard(budget))
    out.mkdir(parents=True, exist_ok=True)
    (out / "SKILL.md").write_text(suggestion.new_skill_md.rstrip() + "\n", encoding="utf-8")
    notes = ["# Suggested rules", ""] + [
        f"- **{r.rule}** ({', '.join(r.conditions)}): {r.reason}" for r in suggestion.rules
    ]
    (out / "SUGGESTIONS.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    typer.echo("\n".join(notes))
    typer.echo(f"\nNew skill: {out / 'SKILL.md'} · suggestion cost ${cost:.4f}")


@command("compare")
def compare(
    before: str = typer.Argument(..., help="Earlier identity (e.g. v1)."),
    after: str = typer.Argument(..., help="Later identity (e.g. v2)."),
    data: Path = typer.Option(Path("data"), "--data"),
) -> None:
    """Compare two identities' records condition by condition (before/after)."""
    a = load_trials(data / "trials" / f"{before}.jsonl")
    b = load_trials(data / "trials" / f"{after}.jsonl")
    if not a or not b:
        raise typer.BadParameter("Both identities need recorded trials.")
    report = build_comparison(before, after, a, b)
    out = data / "records" / f"compare-{before}-{after}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    typer.echo(report)
    typer.echo(f"Saved: {out}")


@command("instructions")
def instructions(skill: Path) -> None:
    """Find rigid instructions and missing escape hatches in a skill (no API call)."""
    from ajantik.instructions import report

    sk = load_skill(skill)
    typer.echo(report(sk.name, sk.files["SKILL.md"].decode("utf-8")))


@command("judge-benchmark")
def judge_benchmark(labels: Path,
                    data: Path = typer.Option(Path("data"), "--data")) -> None:
    """Measure judge versions on a hand-labelled set (no API call)."""
    from ajantik.oracle import measure

    text = measure(labels)
    out = data / "judge" / f"{labels.stem}-benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(text)
    typer.echo(f"Saved: {out}")


@command("reclassify")
def reclassify(
    scenario: Path,
    identity_id: str = typer.Option(..., "--identity",
                                    help="Identity whose records to reclassify."),
    data: Path = typer.Option(Path("data"), "--data"),
    checks: bool = typer.Option(
        False, "--checks",
        help="Also recompute success with the scenario's current checks."),
) -> None:
    """Apply the current judge to recorded trials (texts are kept; no API call; backed up)."""
    import shutil

    from ajantik.trial import ORACLE_VERSION, classify, rescore, right_behaviour

    scen = load_scenario(scenario)
    possible = {f.id: f.success_possible for f in scen.faults}
    store = data / "trials" / f"{identity_id}.jsonl"
    trials = load_trials(store)
    shutil.copy(store, store.with_suffix(f".jsonl.before-judge-v{ORACLE_VERSION}"))
    changed = 0
    for t in trials:
        if checks and t["success"] is not None:
            old = t["success"]
            t["failed_checks"], t["success"] = rescore(scen, t)
            if t["success"] != old:
                typer.echo(f"{t['fault']:28} success {old} → {t['success']} (checks)")
        new = classify(
            t["success"], t["stop"], t["final_text"], scen.failure_words, scen.success_words,
            scen.contract, status_field=scen.status_field,
        )
        right = right_behaviour(new, possible.get(t["fault"], True))
        if new != t.get("outcome") or right != t.get("right"):
            changed += 1
            typer.echo(f"{t['fault']:28} {t.get('outcome') or '-':14} → {new:14} right={right}")
        t["outcome"], t["right"], t["oracle"] = new, right, ORACLE_VERSION
    store.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trials),
                     encoding="utf-8")
    typer.echo(f"{len(trials)} records, {changed} changed (judge v{ORACLE_VERSION}).")


@command("faults")
def faults_cmd(scenario: Path) -> None:
    """Show which fault modules apply to which of the scenario's tools (no API call)."""
    from ajantik.faults import MODULES, applicability

    scen = load_scenario(scenario)
    for name, tools in applicability(scen).items():
        where = ", ".join(tools) if tools else "not applicable"
        typer.echo(f"{name:15} {MODULES[name].description:45} → {where}")
    typer.echo(f"\nConditions in the scenario ({len(scen.faults)}): "
               + ", ".join(f.id for f in scen.faults))


@command("gate")
def gate(
    spec: list[Path] = typer.Argument(..., help="One or more gate definitions (YAML)."),
    data: Path = typer.Option(Path("data"), "--data"),
) -> None:
    """Force a verification gate with variants (no model call, $0)."""
    from ajantik.guards.run import report, run_spec, save

    for sp in spec:
        cases, s = run_spec(sp)
        text = report(sp, s, cases)
        out = save(data / "guards", sp.stem, text, cases)
        typer.echo(text)
        typer.echo(f"Saved: {out}\n")


@command("blind-form")
def blind_form(
    labels: Path,
    form: Path = typer.Option(Path("data/agreement/blind-form.yaml")),
    key: Path = typer.Option(Path("data/agreement/key.yaml"), "--key"),
) -> None:
    """Write a shuffled labelling form that hides the labels (no API call)."""
    from ajantik.agreement import make_blind_form

    form.parent.mkdir(parents=True, exist_ok=True)
    n = make_blind_form(labels, form, key)
    typer.echo(f"{n}-item form: {form}\nKey (do not open until labelling is done): {key}")


@command("agreement")
def agreement(
    labels: Path,
    form: Path = typer.Option(Path("data/agreement/blind-form.yaml")),
    key: Path = typer.Option(Path("data/agreement/key.yaml"), "--key"),
    data: Path = typer.Option(Path("data"), "--data"),
    partial: bool = typer.Option(False, "--partial",
                                 help="Leave unlabelled items out."),
) -> None:
    """Compare a filled blind form with the reference labels: Cohen kappa (no API call)."""
    from ajantik.agreement import agreement_report

    text = agreement_report(labels, form, key, partial=partial)
    out = data / "agreement" / f"{labels.stem}-agreement.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(text)
    typer.echo(f"Saved: {out}")


@command("scenario-from-server")
def scenario_from_server(
    out: Path = typer.Option(..., "--out", help="Scenario file to write (.yaml)."),
    skill: str = typer.Option(..., "--skill", help="Name of the skill folder next to it."),
    task: str = typer.Option(..., "--task", help="The task for the agent."),
    mcp: list[str] = typer.Option(..., "--mcp", help="MCP server command; one per part."),
    sample_replies: bool = typer.Option(
        False, "--sample-replies",
        help="Call each tool inferred as non-writing ONCE with empty arguments for a sample."),
) -> None:
    """Scenario skeleton from a real MCP server's tool schemas (no API call).

    Prefer `ajantik record`: it also captures real replies and the state checks.
    """
    from ajantik.from_server import generate

    text, effects, skipped = generate(list(mcp), skill, task, with_samples=sample_replies)
    if out.exists():
        raise typer.BadParameter(f"File exists; not overwriting: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    typer.echo(f"{len(effects)} tools read → {out}\n")
    typer.echo("INFERRED effect (decides which faults are generated — review it):")
    for name, (effect, basis) in sorted(effects.items()):
        typer.echo(f"  {effect:<4} {name}   ({basis})")
    if skipped:
        typer.echo("\nNo sample reply for:")
        for name, reason in sorted(skipped.items()):
            typer.echo(f"  {name}: {reason}")
    typer.echo("\nYOURS TO FILL: tasks.checks, and output_contract if a program reads the "
               "output. Then: ajantik faults " + str(out))


@command("surface")
def surface(
    mcp: list[str] = typer.Option(..., "--mcp", help="MCP server command; one per part."),
    name: str = typer.Option("", "--name", help="Server name for the report."),
    json_out: Path | None = typer.Option(None, "--json", help="Write the profile as JSON."),
    md_out: Path | None = typer.Option(None, "--md", help="Write the profile as Markdown."),
) -> None:
    """Profile the tool surface an MCP server DECLARES (no model call, $0)."""
    from ajantik.surface import profile, render_markdown, write_profile

    p = profile(list(mcp), name or None)
    write_profile(p, str(json_out) if json_out else None,
                  str(md_out) if md_out else None)
    typer.echo(render_markdown(p))
    if json_out or md_out:
        typer.echo("Written: " + ", ".join(str(x) for x in (json_out, md_out) if x))


@command("surface-batch")
def surface_batch(
    catalogue: Path = typer.Argument(..., help="YAML: server name -> command list."),
    out: Path = typer.Option(..., "--out",
                             help="Directory: one profile per server + index.json."),
) -> None:
    """Profile every MCP server in a catalogue and write an index ($0)."""
    import yaml

    from ajantik.surface import PROFILE_SCHEMA, index_entry, profile, write_profile

    entries = yaml.safe_load(catalogue.read_text(encoding="utf-8")) or {}
    if not isinstance(entries, dict) or not entries:
        raise typer.BadParameter("The catalogue is empty or not 'name: [command, ...]'.")
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], {}
    for name, cmd in entries.items():
        if not isinstance(cmd, list) or not cmd:
            failed[str(name)] = "not a command list"
            typer.echo(f"✗ {name}: not a command list")
            continue
        try:
            p = profile([str(c) for c in cmd], str(name))
        except (OSError, ValueError, RuntimeError) as exc:
            # A connector that will not start is reported, never silently dropped:
            # an index missing its failures would overstate what was covered.
            failed[str(name)] = str(exc)
            typer.echo(f"✗ {name}: {exc}")
            continue
        slug = re.sub(r"[^a-z0-9._-]+", "-", str(name).lower()).strip("-")
        write_profile(p, str(out / f"{slug}.json"), str(out / f"{slug}.md"))
        rows.append({**index_entry(p), "slug": slug})
        c = p["counts"]
        typer.echo(f"✓ {name}: {c['tools']} tools, {c['mutating']} mutating, "
                   f"{c['irreversible']} irreversible")
    index = {
        "schema": PROFILE_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "servers": rows,
        "failed": failed,
        "not_measured": "Behaviour was not measured; only the declared tool surface was read.",
        "no_score": "No composite score or ranking is produced.",
    }
    (out / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"\n{len(rows)} servers profiled, {len(failed)} failed → {out / 'index.json'}")


@command("honesty")
def honesty(
    task: str = typer.Option(..., "--task", help="The task for the agent."),
    mcp: list[str] = typer.Option(..., "--mcp", help="MCP server command; one per part."),
    budget: float = typer.Option(1.00, "--budget", help="Hard spend cap (USD)."),
    reps: int = typer.Option(3, "--reps", help="Trials per fault condition."),
    model: str = "claude-opus-5",
    effort: str = "medium",
    max_tokens: int = 4000,
    instruction: str = typer.Option("", "--instruction",
                                    help="Extra instruction for the agent (default: honesty)."),
    required: list[str] = typer.Option(
        [], "--required",
        help="The task cannot be done without this tool (repeatable). Side-effecting tools "
             "count as required already; use this for read tools."),
    machine_reads: bool = typer.Option(
        False, "--machine-reads",
        help="A PROGRAM reads the output: a machine-readable status line is imposed and the "
             "judge reads that line, not the prose. Silent failures show up here."),
    data: Path = typer.Option(Path("data"), "--data"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Show the plan and worst-case cost; call nothing."),
    out: Path | None = typer.Option(None, "--out", help="Write the report (.md)."),
) -> None:
    """When the tool it relies on breaks, does the agent say so? No scenario needed."""
    from ajantik.honesty import (
        CONTRACT,
        CONTRACT_SUFFIX,
        DEFAULT_INSTRUCTION,
        SANDBOX_TOKEN,
        bench,
        clean_fault,
        conditions,
        harness,
        render,
        scenario,
        summarize,
        synthetic_skill,
        tool_specs,
    )
    from ajantik.mcp import MCPServer

    if model not in PRICES:
        raise typer.BadParameter(f"No price known for model: {model}")
    command_ = list(mcp)
    import tempfile

    with tempfile.TemporaryDirectory() as probe:
        specs = tool_specs([part.replace(SANDBOX_TOKEN, probe) for part in command_])
    if not specs:
        raise typer.BadParameter("The server declared no tools.")
    skill = synthetic_skill(instruction or DEFAULT_INSTRUCTION)
    contract = CONTRACT if machine_reads else None
    prompt = task + (CONTRACT_SUFFIX if machine_reads else "")
    ident = Identity(skill.fingerprint, model, effort, harness(HARNESS, command_))
    worst_call = worst_case_call_usd(model, 6000, max_tokens)
    writes = sum(1 for sp in specs if sp.effect == "write")

    typer.echo(f"Server: {' '.join(command_)}")
    typer.echo(f"{len(specs)} tools declared ({writes} change state) · identity {ident.id}")
    typer.echo("A clean run goes first; faults are applied ONLY to tools that run used, because "
               "breaking an unused tool does not make the task impossible.")
    if dry_run:
        # An upper bound: the clean run decides the real count, and it has not run yet.
        upper = 1 + reps * (len(specs) + writes)
        typer.echo(f"At most {upper} trials (upper bound: every tool counted as used), "
                   f"~${worst_call:.3f} per call · cap ${budget:.2f}")
        return

    import shutil

    import anthropic

    if not any(SANDBOX_TOKEN in part for part in command_):
        typer.echo(f"\nSTOP: no {SANDBOX_TOKEN} in the --mcp arguments. Every trial would use "
                   f"the same directory, one trial's files would be visible to the next, and "
                   f"the measurement would be contaminated. Replace the path you give the "
                   f"server with {SANDBOX_TOKEN}.")
        raise typer.Exit(2)

    client = anthropic.Anthropic()
    guard = BudgetGuard(cap_usd=budget)
    store = data / "trials" / f"{ident.id}.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    trials: list[dict] = []

    def one(scen, fault, rep):
        # A fresh server AND a fresh sandbox per trial. Sharing the directory lets one
        # trial's files be found by the next, so a trial whose writes all failed can
        # still read the right content back and report success -- which is not the
        # agent's error but ours, and it invalidated the first two real runs.
        box = data / "sandboxes" / f"{fault.id.replace(':', '_')}-{rep}"
        if box.exists():
            shutil.rmtree(box)
        box.mkdir(parents=True)
        per_trial = [part.replace(SANDBOX_TOKEN, str(box)) for part in command_]
        with MCPServer(per_trial) as server:
            tools = bench(server, fault)
            res = run_trial(client, skill, scen, scen.tasks[0], fault, ident, rep, guard,
                            max_tokens=max_tokens, tools=tools)
        with store.open("a", encoding="utf-8") as fh:
            fh.write(res.to_json() + "\n")
        trials.append(json.loads(res.to_json()))
        mark = {True: "✓", False: "✗", None: "·"}[res.success]
        line = (f"{mark} {fault.id} #{rep}: {res.stop}, {res.turns} turns, {res.outcome or '—'}, "
                f"${res.cost_usd:.4f} (total ${guard.spent_usd:.4f})")
        if res.error:  # a trial that failed must say why, not just that it failed
            line += f"\n    reason: {res.error}"
        typer.echo(line)
        return res, tools

    if machine_reads:
        typer.echo("Consumer: a PROGRAM. The judge reads the machine-readable line, not the "
                   "warning paragraph — a program never reads the paragraph.")
    clean = clean_fault()
    clean_scen = scenario(command_, prompt, specs, [clean], contract)
    res, tools = one(clean_scen, clean, 1)
    used = dict(tools.calls)   # tool -> how many times the clean run called it
    if res.stop == "budget":
        typer.echo("The cap was reached in the clean run; no fault condition ran.")
        return
    if res.stop == "api_error":
        # Do not blame the task when the model never answered: that is a wrong diagnosis,
        # and spending the remaining budget against a broken API finds nothing.
        typer.echo("\nThe model call failed; no fault condition ran. Reason above.")
        typer.echo("If it is an authentication error, check ANTHROPIC_API_KEY "
                   "(Messages API keys start with sk-ant-api…).")
        return
    if res.stop != "end_turn":
        typer.echo(f"\nThe clean run did not finish ({res.stop}); no fault condition ran.")
        return
    if not used:
        typer.echo("\nThe clean run finished but the agent called no tool; nothing to break. "
                   "Make sure the task needs the tools.")
        return
    typer.echo("\nTools used: " + ", ".join(f"{n}×{c}" for n, c in sorted(used.items())))

    faults = conditions(used, specs, list(required))
    if not faults:
        typer.echo("No fault condition to generate.")
        return
    scen = scenario(command_, prompt, specs, [clean, *faults], contract)
    typer.echo(f"{len(faults)} fault conditions × {reps} reps")
    not_blocking = [n for n in used
                    if n not in required
                    and not any(sp.name == n and sp.effect == "write" for sp in specs)]
    if not_blocking:
        typer.echo(f"Not broken ({', '.join(sorted(not_blocking))}): no side effect, can be "
                   f"worked around. If required, declare it with --required.")
    skipped = [n for n, c in used.items() if c < 2
               and any(sp.name == n and sp.effect == "write" for sp in specs)]
    if skipped:
        # Say what was not tested rather than leaving a quiet gap in coverage.
        typer.echo(f"Session drop skipped ({', '.join(skipped)}): it needs a second call, and "
                   f"this tool was called fewer than 2 times in the clean run.")
    typer.echo("")
    for fault in faults:
        for rep in range(1, reps + 1):
            res, _ = one(scen, fault, rep)
            if res.stop == "budget":
                typer.echo(f"Stopped at the cap: {res.error}")
                break
        else:
            continue
        break

    summary = summarize(trials, contract)
    report = render(summary, model, " ".join(command_), task)
    typer.echo("\n" + report)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        typer.echo(f"Report: {out}")
    typer.echo(f"Trials: {store}  ·  spent ${guard.spent_usd:.4f}")


if __name__ == "__main__":
    app()
