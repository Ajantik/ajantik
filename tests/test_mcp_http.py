"""A remote MCP server (a URL): reached over HTTP, faulted by the same proxy, no secret kept."""

import json
import os
import sys
from pathlib import Path

import pytest

from ajantik import autotest as at
from ajantik.mcp import MCPServer
from ajantik.mcp_http import AuthRequired, HTTPServer, bridge_command, expand
from ajantik.proxy import PHANTOM_DEFAULT, Lab, finish

sys.path.insert(0, str(Path(__file__).parent))
from fixtures_http_mcp import RemoteStore

BEARER = {"Authorization": "Bearer s3cret"}


@pytest.mark.parametrize("sse", [False, True])
def test_a_remote_server_is_listed_and_called_with_its_session(sse):
    with RemoteStore(sse=sse) as remote, HTTPServer(remote.url, BEARER) as s:
        assert s.server_info["serverInfo"]["name"] == "remote-store"
        assert {t["name"] for t in s.list_tools()} == {"store_value", "read_all"}
        assert s.call_tool("store_value", {"field": "a", "value": "1"})["content"][0]["text"] \
            == "Stored (remote)."
        assert remote.store == {"a": "1"}
        assert remote.seen_headers[-1]["Mcp-Session-Id"] == "session-1"
    assert remote.sessions == set()  # closed with DELETE


def test_a_server_that_wants_a_login_says_so():
    with RemoteStore() as remote, pytest.raises(AuthRequired):
        HTTPServer(remote.url, {}).__enter__()


def test_claude_code_variables_are_expanded():
    env = {"TOKEN": "abc"}
    assert expand("Bearer ${TOKEN}", env) == "Bearer abc"
    assert expand("${MISSING:-fallback}/x", env) == "fallback/x"
    assert bridge_command("https://x/mcp")[:3] == ["npx", "-y", "mcp-remote@0.14.3"]


@pytest.fixture
def remote_project(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (proj / ".claude" / "skills" / "notes").mkdir(parents=True)
    (proj / ".claude" / "skills" / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Saves a note.\n---\nSave it.\n")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("AJANTIK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STORE_TOKEN", "s3cret")
    return proj


def test_a_plan_for_a_remote_server_keeps_no_secret(remote_project):
    with RemoteStore() as remote:
        (remote_project / ".mcp.json").write_text(json.dumps({"mcpServers": {"store": {
            "type": "http", "url": remote.url + "?key=hidden",
            "headers": {"Authorization": "Bearer ${STORE_TOKEN}"}}}}))
        plan = at.make_plan("notes", "save a note", remote_project)
    assert plan.servers["store"] == {"type": "http", "url": remote.url, "via": "http",
                                     "headers": ["Authorization"], "vars": ["STORE_TOKEN"]}
    assert at.remote_env(plan) == {"STORE_TOKEN": "s3cret"}  # for the agent's process only
    assert [t["name"] for t in plan.tools["store"] if t["effect"] == "write"] == ["store_value"]
    saved = json.dumps(plan.to_json()) + json.dumps(at.mcp_config(plan, Lab("x")))
    assert "s3cret" not in saved and "hidden" not in saved
    assert f"store: remote, {remote.url}, over HTTP" in at.describe(plan)


def test_the_proxy_faults_a_remote_server(remote_project, tmp_path):
    with RemoteStore() as remote:
        (remote_project / ".mcp.json").write_text(json.dumps({"mcpServers": {"store": {
            "type": "http", "url": remote.url,
            "headers": {"Authorization": "Bearer ${STORE_TOKEN}"}}}}))
        lab = Lab("srv", tmp_path / "home")
        with lab.lock():
            lab._write("schedule.json", ["phantom_success"])
            lab.arm()
        cmd = [sys.executable, "-m", "ajantik.proxy", "--name", "srv", "--remote", "store",
               "--config-cwd", str(remote_project), "--via", "http"]
        with MCPServer(cmd, env=dict(os.environ)) as client:
            assert client.server_info["serverInfo"]["name"] == "remote-store"
            first = client.call_tool("store_value", {"field": "a", "value": "1"})
            assert first["content"][0]["text"] == PHANTOM_DEFAULT
            assert remote.store == {}  # the phantom never reached the remote server
            client.call_tool("store_value", {"field": "a", "value": "1"})
            assert remote.store == {"a": "1"}
    with lab.lock():
        assert finish(lab, "done")["world"] == "repaired"


def test_no_reachable_connector_stops_before_any_run(remote_project):
    (remote_project / ".mcp.json").write_text(json.dumps({"mcpServers": {"store": {
        "type": "http", "url": "http://127.0.0.1:9/mcp", "headers": {"X": "y"}}}}))
    with pytest.raises(ValueError, match="None of the connectors could be reached"):
        at.make_plan("notes", "save a note", remote_project)


def test_a_check_reads_the_system_through_its_connector(remote_project, capsys):
    from ajantik.check import connector, report

    with RemoteStore() as remote:
        remote.store["a"] = "1"
        (remote_project / ".mcp.json").write_text(json.dumps({"mcpServers": {"store": {
            "type": "http", "url": remote.url,
            "headers": {"Authorization": "Bearer ${STORE_TOKEN}"}}}}))
        with connector("store", remote_project) as store:
            assert "read_all" in store.tools()
            held = store.json("read_all")
    report([] if held == {"a": "1"} else ["a is missing"])
    assert json.loads(capsys.readouterr().out) == {"problems": []}
