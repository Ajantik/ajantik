"""The wall without MCP: in-process and over HTTP. Same faults, same record, same verdicts."""

import json
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from ajantik.adapter import Wall
from ajantik.http_wall import READY_PREFIX
from ajantik.rounds import MANIFEST, TASK, WALL_URL, main, run_round, transport
from ajantik.verdicts import verdicts

ROOT = Path(__file__).parent.parent
EN = ROOT / "examples" / "intake-form" / "scenario.yaml"
PHANTOM = "phantom-success:set_field"
HTTP_AGENT = ROOT / "tests" / "stand_in_http_agent.py"
INPROC_AGENT = ROOT / "tests" / "stand_in_inprocess_agent.py"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_in_process_wall_applies_the_fault_and_records_it(tmp_path):
    rec = tmp_path / "s.jsonl"
    with Wall(EN, PHANTOM, rec) as wall:
        text, is_error = wall.call("set_field", {"field": "country", "value": "BG"})
    assert (text, is_error) == ('{"status": "saved"}', False)
    events = rows(rec)
    assert [e["event"] for e in events] == ["session_start", "tool_call", "session_end"]
    assert events[-1]["final_state"] == {}  # the phantom: said saved, stored nothing


def test_tool_lists_come_in_each_sdk_shape():
    wall = Wall(EN)
    assert {t["name"] for t in wall.tools("anthropic")} == {"read_intake", "set_field", "get_field"}
    assert all(t["type"] == "function" and "parameters" in t["function"]
               for t in wall.tools("openai"))
    assert all("inputSchema" in t for t in wall.tools("mcp"))
    with pytest.raises(ValueError):
        wall.tools("langchain")


def test_from_env_says_how_to_start_it(monkeypatch):
    monkeypatch.delenv("AJANTIK_SCENARIO", raising=False)
    with pytest.raises(RuntimeError, match="--in-process"):
        Wall.from_env()


def test_a_closed_session_refuses_more_calls(tmp_path):
    wall = Wall(EN, record=tmp_path / "s.jsonl")
    wall.close()
    with pytest.raises(RuntimeError):
        wall.call("read_intake")


def test_http_wall_serves_openapi_signals_errors_and_writes_the_end_on_sigterm(tmp_path):
    rec = tmp_path / "s.jsonl"
    proc = subprocess.Popen([sys.executable, "-m", "ajantik.http_wall", "--scenario", str(EN),
                             "--fault", "transient-error:set_field", "--record", str(rec)],
                            stdout=subprocess.PIPE, text=True)
    url = proc.stdout.readline()[len(READY_PREFIX):].strip()
    try:
        spec = json.loads(urllib.request.urlopen(f"{url}/openapi.json").read())
        assert set(spec["paths"]) == {"/tools/read_intake", "/tools/set_field", "/tools/get_field"}
        req = urllib.request.Request(f"{url}/tools/set_field", method="POST",
                                     data=json.dumps({"field": "country", "value": "BG"}).encode())
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 503  # the transient error reaches an HTTP client as one
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    assert rows(rec)[-1]["event"] == "session_end"


def test_transport_is_read_from_the_template():
    assert transport(["agent", "{wall}"]) == "mcp"
    assert transport(["agent", "--url", WALL_URL]) == "http"
    assert transport(["agent"], in_process=True) == "in-process"


@pytest.mark.parametrize("mode", ["http", "in-process"])
def test_a_round_without_mcp_yields_the_same_verdicts(tmp_path, mode):
    """A blind agent under phantom success ends with the world wrong; a verifying one fixes
    it by reading back. Exactly what the MCP round shows, through the other transports."""
    out = {}
    for style in ("blind", "verifying"):
        if mode == "http":
            cmd = [sys.executable, str(HTTP_AGENT), "--url", WALL_URL, "--task", TASK,
                   "--style", style]
        else:
            cmd = [sys.executable, str(INPROC_AGENT), "--task", TASK, "--style", style]
        m = run_round(EN, tmp_path / style, cmd, only=[PHANTOM], in_process=mode == "in-process")
        assert m["transport"] == mode
        out[style] = verdicts(EN, {"m": tmp_path / style})["trials"][0]
    assert out["blind"]["world_correct"] is False and out["blind"]["verdict"] == "silent_wrong"
    assert out["verifying"]["world_correct"] is True and out["verifying"]["read_back_calls"] >= 1


def test_cli_accepts_an_in_process_agent_without_a_wall_token(tmp_path):
    main(["--scenario", str(EN), "--out", str(tmp_path), "--fault", PHANTOM, "--in-process",
          "--", sys.executable, str(INPROC_AGENT), "--task", TASK, "--style", "blind"])
    assert json.loads((tmp_path / MANIFEST).read_text())["transport"] == "in-process"
