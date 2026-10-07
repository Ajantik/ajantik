"""Tests for generating a scenario from a real MCP server's tool schemas.

The generator's one guess is `effect`, and that guess decides which fault modules
run. So the tests check the guess in both directions and check that a generated
file actually loads and produces the faults it promises -- a generator whose
output the loader rejects is worse than no generator.
"""

import sys
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from ajantik.from_server import generate, infer_effect, sample_responses, scenario_yaml
from ajantik.mcp import MCPServer
from ajantik.scenario import load_scenario

SERVER = [sys.executable, str(Path(__file__).parent / "fixtures_mcp_server.py")]
SKILL_MD = """---
name: test-skill
description: Writes something and reports that it saved it.
---

Save the given value, then report that you saved it.
"""


@pytest.fixture
def skill_dir(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


class TestInferEffect:
    def test_write_verb_in_name(self):
        effect, basis = infer_effect({"name": "write_file", "description": ""})
        assert effect == "write" and "write" in basis

    def test_write_verb_in_description_when_name_is_neutral(self):
        effect, basis = infer_effect({"name": "mutate", "description": "Will update the record."})
        assert effect == "write" and "in the description" in basis

    def test_read_verb_gives_none_not_read(self):
        # `read` means "return what a write tool stored" in the simulated bench; a real
        # read tool has no such relationship and must not inherit that meaning.
        effect, basis = infer_effect({"name": "list_directory", "description": ""})
        assert effect == "none" and "read verb" in basis

    def test_unknown_tool_defaults_to_non_mutating_and_says_so(self):
        effect, basis = infer_effect({"name": "zyx", "description": "does a thing"})
        assert effect == "none" and "no write verb found" in basis

    def test_name_beats_description_both_ways(self):
        assert infer_effect({"name": "write_thing", "description": "just reads"})[0] == "write"
        # "read the whole store" mentions a store; it does not write to one.
        assert infer_effect({"name": "read_all", "description": "read the whole store"})[0] == "none"


class TestScenarioYaml:
    TOOLS: ClassVar[list[dict]] = [
        {"name": "store_value", "description": "store a value", "inputSchema": {"type": "object"}},
        {"name": "read_all", "description": "read the store", "inputSchema": {"type": "object"}},
    ]

    def test_declares_the_server_command(self):
        data = yaml.safe_load(scenario_yaml(["cmd", "--flag"], self.TOOLS, "skill", "task"))
        assert data["mcp_server"] == ["cmd", "--flag"]

    def test_effects_are_written_and_commented(self):
        text = scenario_yaml(["cmd"], self.TOOLS, "skill", "task")
        assert "# effect inferred:" in text
        data = yaml.safe_load(text)
        effects = {t["name"]: t["effect"] for t in data["tools"]}
        assert effects == {"store_value": "write", "read_all": "none"}

    def test_data_anomaly_is_left_out_without_a_sample(self):
        data = yaml.safe_load(scenario_yaml(["cmd"], self.TOOLS, "skill", "task"))
        assert "data_anomaly" not in data["auto_faults"]

    def test_data_anomaly_is_included_once_a_sample_exists(self):
        text = scenario_yaml(["cmd"], self.TOOLS, "skill", "task",
                             samples={"read_all": '{"a": 1, "b": 2}'})
        data = yaml.safe_load(text)
        assert "data_anomaly" in data["auto_faults"]
        assert data["tools"][1]["response"] == '{"a": 1, "b": 2}'

    def test_descriptions_with_colons_survive(self):
        tools = [{"name": "read_all", "description": "Reads: everything, #1 choice",
                  "inputSchema": {"type": "object"}}]
        data = yaml.safe_load(scenario_yaml(["cmd"], tools, "skill", "task"))
        assert data["tools"][0]["description"] == "Reads: everything, #1 choice"

    def test_checks_are_left_empty_for_a_person_to_fill(self):
        data = yaml.safe_load(scenario_yaml(["cmd"], self.TOOLS, "skill", "task"))
        assert data["tasks"][0]["checks"] == []


class TestAgainstARealServer:
    def test_generate_reads_the_tools(self):
        text, effects, skipped = generate(SERVER, "skill", "task")
        assert set(effects) == {"store_value", "read_all"}
        assert effects["store_value"][0] == "write"
        assert effects["read_all"][0] == "none"
        assert skipped == {}          # sampling was not requested
        assert "mcp_server" in text

    def test_sampling_skips_writers_and_says_why(self):
        with MCPServer(SERVER) as server:
            samples, skipped = sample_responses(
                server, server.list_tools(), {"store_value": "write", "read_all": "none"})
        assert "store_value" not in samples
        assert "side effect" in skipped["store_value"]
        assert samples["read_all"] == "{}"   # the real store, empty

    def test_sampling_never_calls_a_writing_tool(self):
        """Proved from the server's own state, not from a flag."""
        with MCPServer(SERVER) as server:
            sample_responses(server, server.list_tools(), {"store_value": "write", "read_all": "none"})
            after = server.call_tool("read_all", {})
        assert after["content"][0]["text"] == "{}"

    def test_generated_file_loads_and_produces_faults(self, tmp_path, skill_dir):
        text, _, _ = generate(SERVER, skill_dir.name, "Save the value and report it.")
        path = tmp_path / "scenario.yaml"
        path.write_text(text, encoding="utf-8")
        scen = load_scenario(path)
        assert scen.mcp_server == SERVER
        ids = {f.id for f in scen.faults}
        assert "clean" in ids
        assert "phantom-success:store_value" in ids        # only the writing tool
        assert "phantom-success:read_all" not in ids
        assert "premature-read:read_all" in ids          # only the non-writing tool
        assert not any(i.startswith("anomaly-") for i in ids)   # no sample, no anomalies

    def test_empty_server_is_refused(self):
        with pytest.raises(ValueError, match="declared no tools"):
            script = (
                "import json,sys\n"
                "for line in sys.stdin:\n"
                "    m=json.loads(line)\n"
                "    if m.get('id') is not None:\n"
                "        print(json.dumps({'jsonrpc':'2.0','id':m['id'],"
                "'result':{'tools':[]} if m.get('method')=='tools/list' else {}}),flush=True)"
            )
            generate([sys.executable, "-c", script], "skill", "task")
