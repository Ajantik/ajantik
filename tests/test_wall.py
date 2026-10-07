"""The crash-test wall: what an agent we do not own sees over MCP, and what we record."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ajantik.bench import READ_SKILL_FILE
from ajantik.mcp import PROTOCOL_VERSION, MCPServer
from ajantik.scenario import load_scenario
from ajantik.wall import (
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_NAME,
    FaultyServer,
    Transcript,
    pick_fault,
)

EX = Path(__file__).parent.parent / "examples" / "intake-form" / "scenario.yaml"
PHANTOM = "phantom-success:set_field"
TOOLS = ["read_intake", "set_field", "get_field"]
SAVED = {"status": "saved"}


def server(fault_id=PHANTOM):
    scen = load_scenario(EX)
    return FaultyServer(scen, pick_fault(scen, fault_id), Transcript())


def call(srv, name, **arguments):
    reply = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})
    result = reply["result"]
    return result["content"][0]["text"], result["isError"]


def test_tool_list_uses_the_mcp_spelling_of_the_schema():
    srv = server()
    tools = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert [t["name"] for t in tools] == TOOLS
    assert all("inputSchema" in t and "input_schema" not in t for t in tools)


def test_the_lab_harness_affordance_is_never_exposed_or_reachable():
    """read_skill_file belongs to the lab's own harness. Here there is no skill at
    all, so it must be neither advertised nor callable - not even by name."""
    srv = server()
    tools = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert READ_SKILL_FILE not in {t["name"] for t in tools}
    text, is_error = call(srv, READ_SKILL_FILE, path="anything")
    assert is_error and "Unknown tool" in text


def test_session_start_records_whether_success_was_even_possible():
    """A judge that scores the end state without this fails an agent that correctly
    reported it could not be done."""
    assert server(PHANTOM).transcript.entries[0]["success_possible"] is True
    assert server("permanent-error:set_field").transcript.entries[0]["success_possible"] is False
    assert server("session-drop:set_field").transcript.entries[0]["success_possible"] is False


def test_initialize_tells_the_agent_nothing_beyond_the_protocol():
    """The wall must not whisper to the car: no instructions field, and nothing
    that names the fault in effect."""
    result = server().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {}})["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["capabilities"] == {"tools": {}}
    assert "instructions" not in result
    assert PHANTOM not in json.dumps(result) and "phantom" not in json.dumps(result).lower()


def test_phantom_write_is_recorded_on_our_side_not_taken_on_the_agents_word():
    srv = server()
    text, is_error = call(srv, "set_field", field="product_name", value="X")
    assert not is_error and json.loads(text) == SAVED

    # The tool claimed a save. Our state says otherwise, and the transcript keeps both.
    row = srv.transcript.entries[-1]
    assert row["event"] == "tool_call" and row["state_after"] == {}

    read_back, _ = call(srv, "get_field", field="product_name")
    assert json.loads(read_back)["value"] is None

    # phantom_first only lies once: a retry lands, which is what makes the
    # scenario fair to an agent that verifies.
    call(srv, "set_field", field="product_name", value="X")
    assert srv.transcript.entries[-1]["state_after"] == {"product_name": "X"}
    assert json.loads(call(srv, "get_field", field="product_name")[0])["value"] == "X"


def test_session_end_records_the_state_a_state_check_is_judged_against():
    srv = server()
    call(srv, "set_field", field="product_name", value="X")   # phantom: dropped
    call(srv, "set_field", field="country", value="BG")       # lands
    end = srv.finish()
    assert end["event"] == "session_end"
    assert end["final_state"] == {"country": "BG"}
    assert end["calls"] == {"set_field": 2}
    assert end["session_dead"] is False


def test_session_drop_kills_every_later_call():
    srv = server("session-drop:set_field")
    assert call(srv, "set_field", field="a", value="1")[1] is False
    text, is_error = call(srv, "set_field", field="b", value="2")
    assert is_error and "401" in text
    assert call(srv, "get_field", field="a")[1] is True        # the whole session is gone
    assert srv.finish()["session_dead"] is True


def test_clean_fault_tells_no_lies():
    srv = server("clean")
    call(srv, "set_field", field="product_name", value="X")
    assert srv.transcript.entries[-1]["state_after"] == {"product_name": "X"}


@pytest.mark.parametrize("fault_id", [f.id for f in load_scenario(EX).faults])
def test_every_generated_fault_serves_without_crashing(fault_id):
    """A wall that crashes must not be mistaken for an agent that behaved."""
    srv = server(fault_id)
    for name, args in (("read_intake", {}), ("set_field", {"field": "a", "value": "1"}),
                       ("get_field", {"field": "a"})):
        reply = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": name, "arguments": args}})
        assert "result" in reply, reply
    assert not [e for e in srv.transcript.entries if e["event"] == "server_error"]


def test_unknown_fault_id_is_refused_with_the_known_ones():
    scen = load_scenario(EX)
    with pytest.raises(SystemExit) as exc:
        pick_fault(scen, "no-such-fault")
    assert PHANTOM in str(exc.value)


def test_protocol_edges():
    srv = server()
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert srv.initialized is True
    assert srv.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})["result"] == {}
    err = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "resources/list"})["error"]
    assert err["code"] == METHOD_NOT_FOUND


def test_malformed_line_is_answered_not_swallowed(tmp_path):
    import io
    srv = server()
    out = io.StringIO()
    srv.serve(io.StringIO('{"jsonrpc": "2.0"\nnot json at all\n'), out)
    codes = [json.loads(line)["error"]["code"] for line in out.getvalue().splitlines()]
    assert codes == [PARSE_ERROR, PARSE_ERROR]


def test_a_real_mcp_client_can_drive_it_over_stdio(tmp_path):
    """End to end over a pipe, with the transcript written to disk, because an
    external agent will spawn exactly this command."""
    record = tmp_path / "session.jsonl"
    cmd = [sys.executable, "-m", "ajantik.wall", "--scenario", str(EX),
           "--fault", PHANTOM, "--record", str(record)]
    with MCPServer(cmd) as s:
        assert s.server_info["serverInfo"]["name"] == SERVER_NAME
        assert [t["name"] for t in s.list_tools()] == TOOLS
        saved = s.call_tool("set_field", {"field": "product_name", "value": "X"})
        assert json.loads(saved["content"][0]["text"]) == SAVED
        read_back = s.call_tool("get_field", {"field": "product_name"})
        assert json.loads(read_back["content"][0]["text"])["value"] is None

    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert rows[0]["event"] == "session_start" and rows[0]["fault"] == PHANTOM
    saved_row = next(r for r in rows if r.get("tool") == "set_field")
    assert json.loads(saved_row["text"]) == SAVED and saved_row["state_after"] == {}
    assert rows[-1]["event"] == "session_end" and rows[-1]["final_state"] == {}


def test_list_faults_prints_the_scenarios_faults(capsys):
    from ajantik.wall import main

    assert main(["--scenario", str(EX), "--list-faults"]) == 0
    assert PHANTOM in capsys.readouterr().out


def test_serving_path_needs_no_model_sdk():
    """The wall runs on stdlib plus yaml. If it ever needs the Anthropic SDK, a
    third party cannot spawn it cheaply and the archive stops being free to run."""
    code = "import ajantik.wall, sys; assert 'anthropic' not in sys.modules; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).parent.parent), check=False)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_a_wall_stopped_by_a_signal_still_writes_the_final_state(tmp_path, sig):
    """Claude Code stops its MCP servers with a signal instead of closing stdin. Without a
    session_end every trial of such an agent was unjudgeable: seen on the first real run."""
    import signal
    import time

    rec = tmp_path / "s.jsonl"
    proc = subprocess.Popen([sys.executable, "-m", "ajantik.wall", "--scenario", str(EX),
                             "--fault", PHANTOM, "--record", str(rec)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for msg in ({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "set_field",
                            "arguments": {"field": "country", "value": "BG"}}}):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
    time.sleep(0.1)
    proc.send_signal(getattr(signal, sig))
    proc.wait(timeout=10)
    rows = [json.loads(line) for line in rec.read_text().splitlines()]
    assert rows[-1]["event"] == "session_end"
    assert rows[-1]["ended_by"] == sig and rows[-1]["final_state"] == {}
