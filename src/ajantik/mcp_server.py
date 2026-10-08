"""Ajantik as an MCP server: say "test my skill" in Claude Code and get the result in the chat.

Four tools, the same engine as `ajantik test run`:

    list_skills        the skills it can test here
    plan_skill_test    which connectors, which tools write, how many runs -- runs nothing
    start_skill_test   needs the plan id and the user's confirmation; runs in the background
    skill_test_status  progress, then the verdicts and the report

A test takes minutes and MCP clients time a tool call out long before that, so the test runs
as its own process and the agent asks for the status. The confirmation step is the one
question a person must answer: the skill is about to run several times, and every call that
is not faulted changes real things.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

from ajantik import autotest as at
from ajantik.proxy import default_home

INSTRUCTIONS = (
    "Ajantik crash-tests the user's own skills: it runs a skill several times with one tool "
    "fault per run and reports whether the agent told the truth. To test a skill: call "
    "plan_skill_test, show the plan to the user word for word and ask them to confirm, then "
    "call start_skill_test with confirmed=true, then call skill_test_status about every 30 "
    "seconds until it is done, and give the user the summary and the report path.")

TOOLS = [
    {"name": "list_skills",
     "description": "List the Claude Code skills Ajantik can test in a project directory.",
     "inputSchema": {"type": "object", "properties": {
         "cwd": {"type": "string", "description": "Project directory (default: the current one)."}}},
     "annotations": {"readOnlyHint": True}},
    {"name": "plan_skill_test",
     "description": "Plan a crash test of one skill: finds its connectors and which of their "
                    "tools change things. Runs nothing. Show the returned plan to the user and "
                    "get their confirmation before start_skill_test.",
     "inputSchema": {"type": "object", "properties": {
         "skill": {"type": "string", "description": "Skill name, as list_skills shows it."},
         "prompt": {"type": "string",
                    "description": "What the user would type to use the skill, in their words."},
         "cwd": {"type": "string", "description": "Project directory (default: the current one)."},
         "repeat": {"type": "integer", "description": "Runs per fault (default 1)."}},
         "required": ["skill", "prompt"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "start_skill_test",
     "description": "Start a planned test in the background. Only after the user has seen the "
                    "plan and confirmed: the skill runs several times and unfaulted calls make "
                    "real changes.",
     "inputSchema": {"type": "object", "properties": {
         "plan_id": {"type": "string"},
         "confirmed": {"type": "boolean",
                       "description": "True only if the user confirmed this plan."}},
         "required": ["plan_id", "confirmed"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": False}},
    {"name": "skill_test_status",
     "description": "Progress of a started test; when done, the verdicts and the report path.",
     "inputSchema": {"type": "object", "properties": {"plan_id": {"type": "string"}},
                     "required": ["plan_id"]},
     "annotations": {"readOnlyHint": True}},
]


def _plans() -> Path:
    d = default_home() / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lab_name(plan: at.Plan, plan_id: str) -> str:
    return f"auto-{plan.skill}-{plan_id}"


def call(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    cwd = Path(args.get("cwd") or os.getcwd()).expanduser().resolve()
    if name == "list_skills":
        skills = at.list_skills(cwd)
        if not skills:
            return f"No skills found in ~/.claude/skills or {cwd}/.claude/skills.", False
        return "\n".join(f"{s.name} ({s.scope}): {s.description}" for s in skills), False
    if name == "plan_skill_test":
        try:
            plan = at.make_plan(str(args["skill"]), str(args["prompt"]), cwd,
                                repeat=int(args.get("repeat") or 1))
        except ValueError as exc:
            return str(exc), True
        plan_id = secrets.token_hex(4)
        (_plans() / f"{plan_id}.json").write_text(json.dumps(plan.to_json()), encoding="utf-8")
        return (f"plan_id: {plan_id}\n\n{at.describe(plan)}\n\nAsk the user to confirm before "
                "starting."), False
    if name == "start_skill_test":
        path = _plans() / f"{args.get('plan_id')}.json"
        if not path.is_file():
            return "Unknown plan_id. Call plan_skill_test first.", True
        if args.get("confirmed") is not True:
            return "Not started: the user has to confirm the plan first.", True
        plan = at.Plan.from_json(json.loads(path.read_text(encoding="utf-8")))
        if not at.find_claude():
            return "Claude Code (`claude`) is not on PATH; it is needed to run the skill.", True
        log = (_plans() / f"{args['plan_id']}.log").open("w")
        subprocess.Popen(
            [sys.executable, "-m", "ajantik.cli", "test", "run", "--plan-file", str(path),
             "--yes", "--lab-name", _lab_name(plan, str(args["plan_id"]))],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**at.run_env(), "AJANTIK_HOME": str(default_home())},
            start_new_session=True, cwd=plan.cwd)
        return ("Started. It takes a few minutes (one run per fault). Check with "
                "skill_test_status."), False
    if name == "skill_test_status":
        path = _plans() / f"{args.get('plan_id')}.json"
        if not path.is_file():
            return "Unknown plan_id.", True
        plan = at.Plan.from_json(json.loads(path.read_text(encoding="utf-8")))
        status_file = default_home() / "tests" / _lab_name(plan, str(args["plan_id"])) / \
            "status.json"
        if not status_file.is_file():
            tail = (_plans() / f"{args['plan_id']}.log")
            note = tail.read_text(encoding="utf-8")[-600:] if tail.is_file() else ""
            return f"Starting…\n{note}".strip(), False
        st = json.loads(status_file.read_text(encoding="utf-8"))
        lines = [f"{r['run']}. {at.family_name(r['fault'])}: "
                 f"{at.VERDICT_TEXT.get(r['verdict'], r['verdict'])}" for r in st["runs"]]
        if st["state"] == "running":
            head = f"Running: {st['done']} of {st.get('total') or '?'} runs done"
            if st.get("current"):
                head += f", now: {at.family_name(st['current'])}"
            return "\n".join([head, *lines]), False
        if st["state"] == "failed":
            return st.get("error", "The test failed."), True
        return f"Done.\n{st['summary']}\nReport: {st['report']}", False
    return f"Unknown tool: {name}", True


def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method, rpc_id = msg.get("method"), msg.get("id")
    if rpc_id is None:
        return None
    if method == "initialize":
        result: dict[str, Any] = {"protocolVersion": "2025-06-18",
                                  "capabilities": {"tools": {"listChanged": False}},
                                  "serverInfo": {"name": "ajantik", "version": "0.3"},
                                  "instructions": INSTRUCTIONS}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, is_error = call(str(params.get("name")), params.get("arguments") or {})
        except Exception as exc:  # noqa: BLE001 -- a tool error must reach the agent, not kill the server
            text, is_error = f"{type(exc).__name__}: {exc}", True
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": f"method not supported: {method}"}}
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]
    src, out = stdin or sys.stdin, stdout or sys.stdout
    for raw in src:
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        reply = handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            out.flush()
