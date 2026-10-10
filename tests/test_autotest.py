"""Testing a skill automatically: discovery, plan, headless runs, reviewer, MCP tools."""

import json
import sys
from pathlib import Path

import pytest

from ajantik import autotest as at
from ajantik import mcp_server
from ajantik import reviewer as rv
from ajantik.mcp import MCPServer

ROOT = Path(__file__).parent.parent
FAKE = [sys.executable, str(ROOT / "tests" / "fixtures_mcp_server.py")]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A Claude Code config dir with one skill and one stdio server, already wrapped by
    `ajantik test setup` (the discovery has to see through that)."""
    conf = tmp_path / "claude"
    (conf / "skills" / "notes").mkdir(parents=True)
    (conf / "skills" / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Saves a note.\n---\nSave it.\n")
    cwd = tmp_path / "proj"
    cwd.mkdir()
    wrapped = ["proxy", "--name", "store", "--", *FAKE]
    (conf / ".claude.json").write_text(json.dumps({"projects": {str(cwd): {"mcpServers": {
        "store": {"type": "stdio", "command": "/x/ajantik", "args": wrapped, "env": {}},
        "ajantik": {"type": "stdio", "command": "ajantik", "args": ["mcp"], "env": {}},
        "remote": {"type": "http", "url": "https://example.com/mcp"}}}}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(conf))
    monkeypatch.setenv("AJANTIK_HOME", str(tmp_path / "home"))
    return cwd


def test_discovery_finds_skills_and_unwraps_servers(project):
    assert [s.name for s in at.list_skills(project)] == ["notes"]
    servers, notes = at.discover_servers(project)
    assert servers["store"]["command"] == FAKE[0] and servers["store"]["args"] == FAKE[1:]
    assert "remote" not in servers and any("remote" in n for n in notes)
    assert "ajantik" not in servers  # the test runner is never a connector under test


def test_ajantik_is_recognised_however_it_is_started():
    assert at.is_ajantik_itself({"command": "/Users/x/.local/bin/ajantik", "args": ["mcp"]})
    assert at.is_ajantik_itself({"command": "python3", "args": ["-m", "ajantik.cli", "mcp"]})
    assert at.is_ajantik_itself({"command": "python3", "args": ["-m", "ajantik.mcp_server"]})
    assert not at.is_ajantik_itself({"command": "ajantik", "args": ["proxy", "--name", "n",
                                                                    "--", "npx", "srv"]})


def test_the_plan_lists_writing_tools_and_runs_nothing(project):
    plan = at.make_plan("notes", "save a note", project)
    assert plan.write_tools() == ["store: store_value"]
    assert "really happen" in at.describe(plan)
    with pytest.raises(ValueError):
        at.make_plan("nope", "x", project)


def test_faults_follow_what_the_clean_run_called():
    one_write = [{"effect": "read"}, {"effect": "write"}]
    assert at.faults_for(one_write) == ["phantom_success", "transient_error", "permanent_error",
                                        "truncated_reply", "premature_read"]
    assert "session_drop" in at.faults_for(one_write + [{"effect": "write"}])
    assert at.faults_for([{"effect": "read", "is_error": True}]) == []


class FakeReviewer:
    """Reads the prompt the way the real reviewer is asked to, with fixed rules."""
    name = "fake"

    def ask(self, prompt):
        if '"claim"' in prompt:
            message = prompt.split("last message:")[1]
            return '{"claim": "%s", "reason": "fake"}' % (
                "not_done" if "could not" in message else "done")
        return '{"result": "right", "reason": "fake"}'


def scripted(verify):
    """An agent that saves a note through the wrapped server; a careful one reads it back
    and saves again when it is missing."""
    def agent(plan, lab, cwd):
        cfg = at.mcp_config(plan, lab)["mcpServers"]["store"]
        env = {**at.run_env(), **cfg["env"]}
        failed = False
        with MCPServer([cfg["command"], *cfg["args"]], env=env) as s:
            s.list_tools()
            s.call_tool("read_all", {})
            r = s.call_tool("store_value", {"field": "note", "value": "hi"})
            if r.get("isError"):
                r = s.call_tool("store_value", {"field": "note", "value": "hi"})
                failed = bool(r.get("isError"))
            if verify and not failed and \
                    "hi" not in s.call_tool("read_all", {})["content"][0]["text"]:
                s.call_tool("store_value", {"field": "note", "value": "hi"})
        msg = "I could not save the note." if failed else "Saved the note."
        return json.dumps({"type": "result", "result": msg, "num_turns": 3}), ""
    return agent


def test_a_blind_skill_is_caught_and_a_careful_one_is_not(project):
    plan = at.make_plan("notes", "save a note", project)
    blind = at.run_test(plan, scripted(verify=False), FakeReviewer())
    careful = at.run_test(plan, scripted(verify=True), FakeReviewer())
    for status in (blind, careful):
        assert status["state"] == "done" and status["total"] == 6
        assert Path(status["report"]).is_file()
    v_blind = {r["fault"]: r["verdict"] for r in blind["runs"]}
    v_careful = {r["fault"]: r["verdict"] for r in careful["runs"]}
    assert v_blind["clean"] == "correct"
    assert v_blind["phantom_success"] == "silent_wrong"
    assert v_careful["phantom_success"] == "correct"
    assert v_blind["permanent_error"] == "honest_failure"  # it said it could not
    assert v_blind["transient_error"] == "correct"  # it retried
    assert "SILENT WRONG" in blind["summary"] and "Saved the note" in blind["summary"]


def test_a_skill_that_never_touches_the_connectors_is_not_tested(project):
    plan = at.make_plan("notes", "save a note", project)

    def idle(plan, lab, cwd):
        return json.dumps({"type": "result", "result": "Done.", "num_turns": 1}), ""
    status = at.run_test(plan, idle, FakeReviewer())
    assert status["state"] == "failed" and "Nothing was tested" in status["error"]


def test_the_reviewer_answer_is_parsed_or_left_unsure():
    class Says:
        name = "s"

        def __init__(self, text):
            self.text = text

        def ask(self, prompt):
            return self.text
    assert rv.claim(Says('ok {"claim": "done", "reason": "r"}'), "t", "m").value == "done"
    assert rv.claim(Says("no json"), "t", "m").value == "unsure"
    assert rv.claim(Says('{"claim": "done"}'), "t", "").value == "unsure"  # no message


def test_mcp_tools_plan_then_refuse_to_start_unconfirmed(project, monkeypatch):
    monkeypatch.chdir(project)
    listed = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert [t["name"] for t in listed["result"]["tools"]] == [
        "list_skills", "plan_skill_test", "start_skill_test", "skill_test_status"]
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize"})
    assert "confirm" in init["result"]["instructions"]
    text, err = mcp_server.call("list_skills", {})
    assert "notes (user)" in text and not err
    text, err = mcp_server.call("plan_skill_test", {"skill": "notes", "prompt": "save a note"})
    assert not err and "store_value" in text
    plan_id = text.split("plan_id: ")[1].split()[0]
    text, err = mcp_server.call("start_skill_test", {"plan_id": plan_id, "confirmed": False})
    assert err and "confirm" in text


def test_inputs_the_skill_reads_as_files_are_copied_fresh_into_every_run(tmp_path):
    project = tmp_path / "project"
    (project / "inbox").mkdir(parents=True)
    (project / "inbox" / "leads.json").write_text('[{"name": "A"}]')
    (project / "notes.txt").write_text("hello")
    plan = at.Plan("s", "p", str(project), {}, {}, inputs=["inbox", "notes.txt"])
    lab = at.Lab("inputs-test", tmp_path / "home")
    box = at.sandbox(lab, plan)
    assert (box / "inbox" / "leads.json").read_text() == '[{"name": "A"}]'
    assert (box / "notes.txt").read_text() == "hello"
    (box / "inbox" / "leads.json").unlink()            # a run that "processed" its inbox
    (box / "notes.txt").write_text("changed")
    at.copy_inputs(plan, box)                          # the next run starts as the first did
    assert (box / "inbox" / "leads.json").exists()
    assert (box / "notes.txt").read_text() == "hello"
    assert "Copied into each run as files: inbox, notes.txt" in at.describe(plan)


def test_the_cost_of_each_run_is_read_from_claude_code_output():
    out = '{"type":"system"}\n{"type":"result","result":"done","total_cost_usd":0.21734}\n'
    assert at.run_cost(out) == 0.2173
    assert at.run_cost("not json") is None
    text = at.summary_text([{"fault": "clean", "verdict": "correct", "cost_usd": 0.2},
                            {"fault": "transient_error", "verdict": "correct", "cost_usd": 0.3}])
    assert "$0.50 in all ($0.20–$0.30 per run)" in text
