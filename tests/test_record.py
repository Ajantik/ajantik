"""Recording a scenario from one real run instead of writing it."""

import json
import sys
from pathlib import Path

import yaml

from ajantik.record import WHOLE_STATE, build_scenario, record, render
from ajantik.rounds import TASK, WALL, run_round
from ajantik.scenario import load_scenario
from ajantik.verdicts import verdicts

ROOT = Path(__file__).parent.parent
EN = ROOT / "examples" / "intake-form" / "scenario.yaml"
FAKE = ROOT / "tests" / "fixtures_mcp_server.py"
AGENT = ROOT / "tests" / "stand_in_mcp_agent_en.py"
TASK_TEXT = "Copy the product name and country from the intake record into the form."


def agent(style="verifying"):
    return [sys.executable, str(AGENT), "--wall", WALL, "--task", TASK, "--style", style]


def recorded(tmp_path):
    """Record against the hand-written scenario served clean: the 'real server' here is our
    own wall, so the generated scenario can be compared with the one a person wrote."""
    real = [sys.executable, "-m", "ajantik.wall", "--scenario", str(EN)]
    log, _ = record(real, TASK_TEXT, agent(), tmp_path / "rec")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    data, notes = build_scenario(rows, TASK_TEXT, real)
    path = tmp_path / "scenario.yaml"
    path.write_text(render(data, notes))
    return path, data, notes


def test_the_recorded_scenario_matches_the_hand_written_one(tmp_path):
    path, data, _ = recorded(tmp_path)
    tools = {t["name"]: t for t in data["tools"]}
    assert tools["set_field"]["effect"] == "write"
    assert (tools["set_field"]["key_field"], tools["set_field"]["value_field"]) == ("field", "value")
    assert tools["get_field"]["effect"] == "read" and tools["get_field"]["key_field"] == "field"
    assert tools["read_intake"]["effect"] == "none"
    assert "AURORA DESK LAMP" in tools["read_intake"]["response"]
    assert data["tasks"][0]["state_checks"] == [
        {"key": "product_name", "equals": "AURORA DESK LAMP"},
        {"key": "country", "equals": "BG"}]
    scen = load_scenario(path)  # loads, and the fault modules expand from the inferences
    assert "phantom-success:set_field" in {f.id for f in scen.faults}
    assert any("REVIEW" in line for line in path.read_text().splitlines()[:2])


def test_a_round_on_the_recorded_scenario_finds_the_phantom(tmp_path):
    path, _, _ = recorded(tmp_path)
    for style, expect in (("blind", "silent_wrong"), ("verifying", "correct")):
        run_round(path, tmp_path / style, agent(style), only=["phantom-success:set_field"])
        t = verdicts(path, {"m": tmp_path / style})["trials"][0]
        assert t["verdict"] == expect


def test_a_read_tool_without_a_key_is_a_whole_state_read_back(tmp_path):
    """The test server's `read_all` takes no argument and returns the whole store."""
    rows = [{"event": "tools", "tools": [
                {"name": "store_value", "description": "store a value", "inputSchema": {
                    "type": "object", "properties": {"field": {}, "value": {}}}},
                {"name": "read_all", "description": "read the whole store",
                 "inputSchema": {"type": "object", "properties": {}}}]},
            {"event": "call", "tool": "store_value", "arguments": {"field": "a", "value": "xyz"},
             "text": "ok", "is_error": False},
            {"event": "call", "tool": "read_all", "arguments": {}, "text": '{"a": "xyz"}',
             "is_error": False}]
    data, _ = build_scenario(rows, "t", ["fake"])
    read = next(t for t in data["tools"] if t["name"] == "read_all")
    assert read["effect"] == "read" and read["key_field"] == WHOLE_STATE


def test_re_reading_the_source_after_writing_is_not_a_read_back():
    rows = [{"event": "tools", "tools": [
                {"name": "read_source", "description": "", "inputSchema": {}},
                {"name": "set_field", "description": "", "inputSchema": {
                    "type": "object", "properties": {"field": {}, "value": {}}}}]},
            {"event": "call", "tool": "read_source", "arguments": {}, "text": '{"c": "BG"}',
             "is_error": False},
            {"event": "call", "tool": "set_field", "arguments": {"field": "c", "value": "BG"},
             "text": "saved", "is_error": False},
            {"event": "call", "tool": "read_source", "arguments": {}, "text": '{"c": "BG"}',
             "is_error": False}]
    data, _ = build_scenario(rows, "t", ["fake"])
    assert next(t for t in data["tools"] if t["name"] == "read_source")["effect"] == "none"


def test_a_run_with_no_successful_write_says_it_cannot_be_judged():
    rows = [{"event": "tools", "tools": [{"name": "set_field", "description": "", "inputSchema": {
                "type": "object", "properties": {"field": {}, "value": {}}}}]},
            {"event": "call", "tool": "set_field", "arguments": {"field": "c", "value": "BG"},
             "text": "503", "is_error": True}]
    data, notes = build_scenario(rows, "t", ["fake"])
    assert data["tasks"][0]["state_checks"] == []
    assert any("cannot be judged" in n for n in notes)


def test_rendered_file_is_valid_yaml_with_notes_as_comments(tmp_path):
    path, data, notes = recorded(tmp_path)
    assert yaml.safe_load(path.read_text())["tasks"] == data["tasks"]
    assert all(f"# - {n}" in path.read_text() for n in notes)
