"""Tests for the real-MCP bench.

The server is the stateful fake in `fixtures_mcp_server.py`, so "was the call
forwarded" is not asserted on a flag the code sets itself -- it is read back out of
the server's own store. A fault that claims to suppress a write but forwards it
would pass a flag-based test and fail these.
"""

import json
import sys
from pathlib import Path

import pytest

from ajantik.mcp import MCPError, MCPServer, MCPTools, check_supported
from ajantik.scenario import Check, Fault, Scenario, StateCheck, Task

SERVER = [sys.executable, str(Path(__file__).parent / "fixtures_mcp_server.py")]


@pytest.fixture
def server():
    with MCPServer(SERVER) as s:
        yield s


def store(server: MCPServer) -> dict:
    """What the server really holds, independent of what the agent was shown."""
    result = server.call_tool("read_all", {})
    return json.loads(result["content"][0]["text"])


def test_definitions_come_from_the_server(server):
    tools = MCPTools(server, Fault(id="none"))
    names = {d["name"] for d in tools.definitions()}
    assert names == {"store_value", "read_all"}
    assert all("input_schema" in d for d in tools.definitions())


def test_tool_filter_narrows_the_surface(server):
    tools = MCPTools(server, Fault(id="none"), tool_filter=["read_all"])
    assert [d["name"] for d in tools.definitions()] == ["read_all"]
    text, is_error = tools.execute("store_value", {"field": "a", "value": "1"})
    assert is_error and "Unknown tool" in text


def test_clean_run_forwards_and_really_writes(server):
    tools = MCPTools(server, Fault(id="none"))
    text, is_error = tools.execute("store_value", {"field": "a", "value": "1"})
    assert not is_error and "(real)" in text
    assert store(server) == {"a": "1"}


def test_phantom_does_not_reach_the_server(server):
    """The point of a phantom save: the agent is told it saved, nothing is stored.

    Forwarding the write and then reporting success would measure nothing -- the
    claim would be true.
    """
    tools = MCPTools(server, Fault(id="f", kind="phantom_first", tool="store_value"))
    text, is_error = tools.execute("store_value", {"field": "a", "value": "1"})
    assert not is_error
    assert "(real)" not in text          # the server's own wording never appeared
    assert store(server) == {}           # and nothing was stored
    assert tools.log[0]["forwarded_to_server"] is False


def test_phantom_first_only_affects_the_first_call(server):
    tools = MCPTools(server, Fault(id="f", kind="phantom_first", tool="store_value"))
    tools.execute("store_value", {"field": "a", "value": "1"})
    tools.execute("store_value", {"field": "b", "value": "2"})
    assert store(server) == {"b": "2"}

def test_phantom_always_suppresses_every_write(server):
    tools = MCPTools(server, Fault(id="f", kind="phantom_always", tool="store_value"))
    tools.execute("store_value", {"field": "a", "value": "1"})
    tools.execute("store_value", {"field": "b", "value": "2"})
    assert store(server) == {}


def test_error_does_not_reach_the_server(server):
    """A fabricated failure must not let the side effect happen, or the agent is
    right to say it failed while the write actually landed."""
    tools = MCPTools(server, Fault(id="f", kind="error_always", tool="store_value", message="503"))
    text, is_error = tools.execute("store_value", {"field": "a", "value": "1"})
    assert is_error and text == "503"
    assert store(server) == {}


def test_session_drop_kills_later_calls_including_other_tools(server):
    tools = MCPTools(server, Fault(id="f", kind="session_drop", tool="store_value", after=1,
                                  message="401 session expired"))
    tools.execute("store_value", {"field": "a", "value": "1"})      # allowed
    text, is_error = tools.execute("store_value", {"field": "b", "value": "2"})
    assert is_error and "401" in text
    assert tools.session_dead
    text, is_error = tools.execute("read_all", {})
    assert is_error and "401" in text
    assert store(server) == {"a": "1"}


def test_malformed_forwards_then_cuts(server):
    server.call_tool("store_value", {"field": "a", "value": "a-long-value"})
    tools = MCPTools(server, Fault(id="f", kind="malformed_first", tool="read_all"))
    full = json.dumps(store(server), ensure_ascii=False)
    text, is_error = tools.execute("read_all", {})
    assert not is_error
    assert text and len(text) < len(full)
    assert tools.log[-1]["forwarded_to_server"] is True


def test_empty_forwards_then_empties(server):
    server.call_tool("store_value", {"field": "a", "value": "1"})
    tools = MCPTools(server, Fault(id="f", kind="empty_first", tool="read_all"))
    text, _ = tools.execute("read_all", {})
    assert "1" not in text
    second, _ = tools.execute("read_all", {})
    assert "1" in second          # only the first read looks empty


def test_replace_substitutes_the_declared_response(server):
    tools = MCPTools(server, Fault(id="f", kind="replace", tool="read_all",
                                  response='{"a": "anomaly"}'))
    text, _ = tools.execute("read_all", {})
    assert text == '{"a": "anomaly"}'


def test_faults_only_hit_their_own_tool(server):
    tools = MCPTools(server, Fault(id="f", kind="error_always", tool="read_all", message="503"))
    _, is_error = tools.execute("store_value", {"field": "a", "value": "1"})
    assert not is_error and store(server) == {"a": "1"}


def test_log_records_what_was_asked_and_done(server):
    tools = MCPTools(server, Fault(id="f", kind="phantom_first", tool="store_value"))
    tools.execute("store_value", {"field": "a", "value": "1"})
    tools.execute("read_all", {})
    assert [e["tool"] for e in tools.log] == ["store_value", "read_all"]
    assert [e["fault_applied"] for e in tools.log] == [True, False]


def test_state_is_not_claimed_to_be_known(server):
    """The bench must not offer a state dict it cannot fill; the oracle would pass
    on an empty one."""
    tools = MCPTools(server, Fault(id="none"))
    tools.execute("store_value", {"field": "a", "value": "1"})
    assert tools.state == {}


def _scenario(faults):
    return Scenario(skill_path=Path("."), tasks=[Task(id="t", prompt="p", checks=[])],
                    tools=[], faults=faults)


def test_check_supported_refuses_state_check_scenarios():
    scen = _scenario([Fault(id="f1", kind="phantom_first", tool="store_value",
                            state_checks=[StateCheck(key="a", equals="1")])])
    with pytest.raises(ValueError, match="state_checks"):
        check_supported(scen)


def test_check_supported_allows_text_only_scenarios():
    check_supported(_scenario([Fault(id="f1", kind="phantom_first", tool="store_value",
                                     checks=[Check(kind="contains", value="x")])]))


def test_unreachable_server_fails_loudly():
    with pytest.raises((MCPError, OSError)), MCPServer([sys.executable, "-c", "raise SystemExit(1)"]):
        pass
