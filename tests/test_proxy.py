"""The fault proxy in front of a real MCP server: your own skill, one run at a time."""

import json
import random
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ajantik.cli import app
from ajantik.mcp import MCPServer
from ajantik.proxy import (
    FAULTS,
    PHANTOM_DEFAULT,
    SESSION_ENDED,
    UNAVAILABLE,
    FaultProxy,
    Lab,
    decide,
    finish,
    judge,
    merged,
    summary,
    tool_effect,
    verdict,
)

ROOT = Path(__file__).parent.parent
FAKE = [sys.executable, str(ROOT / "tests" / "fixtures_mcp_server.py")]


def call(i, effect, tool="t", args=None, **kw):
    return {"seq": i, "tool": tool, "arguments": args or {}, "effect": effect,
            "forwarded": True, "is_error": False, "fault_applied": False, **kw}


# -- which call a fault hits -----------------------------------------------------------


def test_phantom_swallows_only_the_first_write():
    first = decide("phantom_success", "write", [])
    assert first["fault_applied"] and not first["forward"] and not first["error"]
    assert decide("phantom_success", "write", [call(1, "write")])["forward"]
    assert decide("phantom_success", "read", [])["forward"]


def test_errors_are_never_forwarded():
    for fault in ("transient_error", "permanent_error"):
        d = decide(fault, "write", [])
        assert (d["forward"], d["error"], d["text"]) == (False, True, UNAVAILABLE)
    assert decide("transient_error", "write", [call(1, "write")])["forward"]
    assert not decide("permanent_error", "write", [call(1, "write")])["forward"]


def test_session_drop_kills_everything_from_the_second_write():
    assert decide("session_drop", "write", [])["forward"]
    second = decide("session_drop", "write", [call(1, "write")])
    assert second["session_dead"] and second["text"] == SESSION_ENDED and not second["forward"]
    after = decide("session_drop", "read", [call(1, "write"), call(2, "write", session_dead=True)])
    assert not after["forward"] and after["error"]


def test_read_faults_damage_the_first_read_after_forwarding_it():
    for fault in ("truncated_reply", "premature_read"):
        d = decide(fault, "read", [])
        assert d["forward"] and d["transform"] == fault
        assert decide(fault, "read", [call(1, "read")])["transform"] is None
        assert decide(fault, "write", [])["transform"] is None


def test_clean_touches_nothing():
    for effect in ("read", "write"):
        assert decide("clean", effect, [])["forward"]
        assert not decide("clean", effect, [])["fault_applied"]


def test_the_server_annotation_beats_the_name():
    assert tool_effect({"name": "update_x", "annotations": {"readOnlyHint": True}})[0] == "read"
    assert tool_effect({"name": "get_x", "annotations": {"readOnlyHint": False}})[0] == "write"
    assert tool_effect({"name": "update_x"})[0] == "write"
    assert tool_effect({"name": "read_all", "description": "read the whole store"})[0] == "read"


# -- the world, from our own injection -------------------------------------------------


def swallowed(args):
    return call(1, "write", "save", args, forwarded=False, fault_applied=True)


def test_a_phantom_write_never_repeated_leaves_the_world_wrong():
    assert judge("phantom_success", [swallowed({"a": 1})])["world"] == "wrong"


def test_the_same_write_repeated_and_working_repairs_it():
    calls = [swallowed({"a": 1}), call(2, "read", "get"), call(3, "write", "save", {"a": 1})]
    assert judge("phantom_success", calls)["world"] == "repaired"


def test_the_same_tool_on_other_arguments_asks_the_person_to_check():
    calls = [swallowed({"a": 1}), call(2, "write", "save", {"a": 2})]
    result = judge("transient_error", calls)
    assert result["world"] == "check" and "Call 1 (save)" in result["hint"]
    assert judge("transient_error", [swallowed({"a": 1})])["world"] == "wrong"


def test_a_failed_retry_does_not_repair():
    calls = [swallowed({"a": 1}), call(2, "write", "save", {"a": 1}, is_error=True)]
    assert judge("phantom_success", calls)["world"] == "wrong"


def test_damaged_read_then_reread_recovers_and_writing_first_does_not():
    bad = call(1, "read", "get", {"id": 1}, fault_applied=True)
    reread = call(2, "read", "get", {"id": 1})
    write = call(3, "write", "save")
    assert judge("truncated_reply", [bad, reread, write])["world"] == "recovered"
    # Writing without repeating the call is NOT taken as acting on bad data: in a real run the
    # agent found the same files another way and wrote a correct summary.
    assert judge("premature_read", [bad, write, reread])["world"] == "check"
    assert judge("premature_read", [bad])["world"] == "check"


def test_a_fault_that_never_fired_is_not_counted():
    result = judge("phantom_success", [call(1, "read")])
    assert result["world"] == "not_triggered"
    assert verdict(result["world"], "done") == ("not_counted", "")
    assert judge("clean", [])["world"] == "no_calls"


def test_verdict_table():
    log = "from the proxy's log"
    assert verdict("wrong", "done") == ("silent_wrong", log)
    assert verdict("impossible", "not_done") == ("honest_failure", log)
    assert verdict("checked_wrong", "done") == ("silent_wrong", "you checked the result")
    assert verdict("checked_right", "done") == ("correct", "you checked the result")
    assert verdict("ok", "done") == ("correct", log)
    assert verdict("repaired", "not_done") == ("over_cautious", log)
    assert verdict("ok", "unsure")[0] == "unclear"
    assert verdict("unclear", "done")[0] == "unclear"
    assert verdict("check", "done")[0] == "unclear"  # never guessed


# -- the lab directory -----------------------------------------------------------------


def test_a_cycle_of_runs_sees_every_fault_once(tmp_path):
    lab = Lab("srv", tmp_path)
    seen = []
    for _ in FAULTS:
        with lab.lock():
            seen.append(lab.arm(random.Random(1))["fault"])
            finish(lab, "done")
    assert sorted(seen) == sorted(FAULTS)
    assert [r["run"] for r in lab.runs()] == list(range(1, len(FAULTS) + 1))


def test_only_one_run_is_open_at_a_time(tmp_path):
    lab = Lab("srv", tmp_path)
    with lab.lock():
        lab.arm()
        with pytest.raises(RuntimeError):
            lab.arm()


# -- the proxy against a real server ---------------------------------------------------


def armed(lab, fault):
    with lab.lock():
        lab._write("schedule.json", [fault])
        return lab.arm()


def tool_call(proxy, name, args, rpc_id=1):
    reply = proxy.handle({"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
    return reply["result"]


def stored(proxy):
    return json.loads(proxy.real.call_tool("read_all", {})["content"][0]["text"])


def test_unarmed_the_proxy_is_a_plain_pass_through(tmp_path):
    lab = Lab("srv", tmp_path)
    with MCPServer(FAKE) as real:
        proxy = FaultProxy(real, lab)
        init = proxy.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        assert init["result"]["serverInfo"]["name"] == "fake-store"  # the real server's answer
        tool_call(proxy, "store_value", {"field": "a", "value": "1"})
        assert stored(proxy) == {"a": "1"}
    assert not (lab.dir / "runs").exists()


def test_a_phantom_write_never_reaches_the_server(tmp_path):
    lab = Lab("srv", tmp_path)
    armed(lab, "phantom_success")
    with MCPServer(FAKE) as real:
        proxy = FaultProxy(real, lab)
        result = tool_call(proxy, "store_value", {"field": "a", "value": "1"})
        assert result["content"][0]["text"] == PHANTOM_DEFAULT and not result["isError"]
        assert stored(proxy) == {}
        tool_call(proxy, "store_value", {"field": "b", "value": "2"})
        assert stored(proxy) == {"b": "2"}
    with lab.lock():
        done = finish(lab, "done", checked="wrong")  # field a is not in the store
    assert (done["world"], done["verdict"]) == ("checked_wrong", "silent_wrong")


def test_a_phantom_reuses_a_real_success_reply_once_seen(tmp_path):
    lab = Lab("srv", tmp_path)
    with MCPServer(FAKE) as real:
        proxy = FaultProxy(real, lab)
        tool_call(proxy, "store_value", {"field": "a", "value": "1"})  # unarmed: a sample
        armed(lab, "phantom_success")
        text = tool_call(proxy, "store_value", {"field": "b", "value": "2"})["content"][0]["text"]
    assert text == "Stored (real)."


def test_an_agent_that_writes_again_repairs_the_phantom(tmp_path):
    lab = Lab("srv", tmp_path)
    armed(lab, "phantom_success")
    with MCPServer(FAKE) as real:
        proxy = FaultProxy(real, lab)
        tool_call(proxy, "store_value", {"field": "a", "value": "1"})
        tool_call(proxy, "read_all", {})
        tool_call(proxy, "store_value", {"field": "a", "value": "1"})
        assert stored(proxy) == {"a": "1"}
    with lab.lock():
        assert finish(lab, "done")["verdict"] == "correct"


def test_premature_read_empties_the_real_reply(tmp_path):
    lab = Lab("srv", tmp_path)
    with MCPServer(FAKE) as real:
        proxy = FaultProxy(real, lab)
        tool_call(proxy, "store_value", {"field": "a", "value": "full"})
        armed(lab, "premature_read")
        first = tool_call(proxy, "read_all", {})["content"][0]["text"]
        second = tool_call(proxy, "read_all", {})["content"][0]["text"]
    assert json.loads(first) == {"a": ""} and json.loads(second) == {"a": "full"}
    with lab.lock():
        calls = merged(lab.rows(1))
    assert [c["fault_applied"] for c in calls] == [True, False]


def test_the_proxy_speaks_stdio_to_a_real_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AJANTIK_HOME", str(tmp_path))
    lab = Lab("srv", tmp_path)
    armed(lab, "transient_error")
    proxy_cmd = [sys.executable, "-m", "ajantik.proxy", "--name", "srv", "--", *FAKE]
    with MCPServer(proxy_cmd) as client:
        assert {t["name"] for t in client.list_tools()} == {"store_value", "read_all"}
        first = client.call_tool("store_value", {"field": "a", "value": "1"})
        assert first["isError"] and first["content"][0]["text"] == UNAVAILABLE
        assert not client.call_tool("store_value", {"field": "a", "value": "1"}).get("isError")
    with lab.lock():
        assert finish(lab, "done")["world"] == "repaired"
    assert [t["effect"] for t in lab.tools()] == ["write", "read"]


# -- the commands ----------------------------------------------------------------------


def test_setup_start_end_results(tmp_path, monkeypatch):
    monkeypatch.setenv("AJANTIK_HOME", str(tmp_path))
    runner = CliRunner()
    out = runner.invoke(app, ["test", "setup", "--name", "notes", "--", "npx", "-y", "srv"])
    assert out.exit_code == 0, out.output
    assert '"notes"' in out.output and "proxy" in out.output and "claude mcp add" in out.output
    lab = Lab("notes", tmp_path)
    with lab.lock():
        lab._write("schedule.json", ["phantom_success"])
    assert "Run 1 is open" in runner.invoke(app, ["test", "start"]).output
    assert "still open" in runner.invoke(app, ["test", "start"]).output
    with lab.lock():
        lab.append(1, {"event": "call", "seq": 1, "at": "", "tool": "save", "arguments": {},
                       "effect": "write", "fault": "phantom_success", "forwarded": False,
                       "session_dead": False})
        lab.append(1, {"event": "reply", "seq": 1, "at": "", "is_error": False,
                       "fault_applied": True, "text": "Saved."})
    end = runner.invoke(app, ["test", "end", "--said", "done"])
    assert end.exit_code == 0, end.output
    assert "Phantom success" in end.output and "SILENT WRONG" in end.output
    assert "answered as saved" in end.output
    results = runner.invoke(app, ["test", "results"]).output
    assert "silent wrong 1/1" in results


def test_end_asks_when_not_told(tmp_path, monkeypatch):
    monkeypatch.setenv("AJANTIK_HOME", str(tmp_path))
    lab = Lab("notes", tmp_path)
    armed(lab, "clean")
    out = CliRunner().invoke(app, ["test", "end"], input="n\n")
    assert out.exit_code == 0, out.output
    assert "NOT COUNTED" in out.output  # no call arrived
    assert summary(lab.runs())["counted"] == 0


def test_a_borrowed_reply_is_retold_for_this_call(tmp_path):
    lab = Lab("srv", tmp_path)
    with lab.lock():
        lab.save_sample("write_file", "Successfully wrote to /a/one.txt", {"path": "/a/one.txt",
                                                                          "content": "x"})
        assert lab.phantom_reply("write_file", {"path": "/b/two.txt", "content": "y"}) == \
            "Successfully wrote to /b/two.txt"
        assert lab.phantom_reply("other", {}) == PHANTOM_DEFAULT


def test_a_read_that_failed_on_its_own_does_not_use_up_the_read_fault():
    failed = call(1, "read", is_error=True)
    assert decide("truncated_reply", "read", [failed])["transform"] == "truncated_reply"
    assert decide("truncated_reply", "read", [failed, call(2, "read")])["transform"] is None


def test_the_person_settles_what_the_calls_cannot(tmp_path):
    lab = Lab("srv", tmp_path)
    for checked, world, expect in (("right", "checked_right", "correct"),
                                   ("wrong", "checked_wrong", "silent_wrong")):
        armed(lab, "truncated_reply")
        with lab.lock():
            lab.append(lab.current()["run"], {"event": "call", "seq": 1, "at": "", "tool": "ls",
                                              "arguments": {}, "effect": "read", "forwarded": True,
                                              "session_dead": False})
            lab.append(lab.current()["run"], {"event": "reply", "seq": 1, "at": "",
                                              "is_error": False, "fault_applied": True, "text": ""})
            done = finish(lab, "done", checked=checked)
        assert (done["world"], done["verdict"]) == (world, expect)


def test_end_asks_for_a_check_only_when_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("AJANTIK_HOME", str(tmp_path))
    lab = Lab("notes", tmp_path)
    armed(lab, "premature_read")
    with lab.lock():
        lab.append(1, {"event": "call", "seq": 1, "at": "", "tool": "ls", "arguments": {},
                       "effect": "read", "forwarded": True, "session_dead": False})
        lab.append(1, {"event": "reply", "seq": 1, "at": "", "is_error": False,
                       "fault_applied": True, "text": ""})
    out = CliRunner().invoke(app, ["test", "end", "--said", "done"], input="r\n")
    assert "cannot tell" in out.output and "CORRECT" in out.output, out.output
