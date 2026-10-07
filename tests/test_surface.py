"""Tests for the zero-cost tool-surface profile.

Two things are checked hardest. First, every false negative direction: a profile
that misses a target parameter or an irreversible operation makes a connector look
tamer than it is, which is the one error that matters here. Second, that no score
or grade can appear in the output -- the absence is a product decision, so it is
asserted, not assumed.
"""

import json
import sys
from pathlib import Path

import pytest

from ajantik.surface import (
    PROFILE_SCHEMA,
    index_entry,
    profile,
    profile_tool,
    render_markdown,
    write_profile,
)

SERVER = [sys.executable, str(Path(__file__).parent / "fixtures_mcp_server.py")]


def tool(name, props=None, required=None, description=""):
    schema = {"type": "object", "properties": props or {}}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "inputSchema": schema}


class TestProfileTool:
    def test_irreversible_operation_is_flagged(self):
        p = profile_tool(tool("delete_entities"))
        assert p["irreversible"] and "delete" in p["irreversible_basis"]
        assert p["effect"] == "write"

    def test_creating_is_mutating_but_not_irreversible(self):
        p = profile_tool(tool("create_entities"))
        assert p["effect"] == "write"
        assert not p["irreversible"]

    def test_reading_is_neither(self):
        p = profile_tool(tool("read_graph"))
        assert p["effect"] == "none" and not p["irreversible"]

    def test_camelcase_safeguard_param_is_found(self):
        p = profile_tool(tool("edit_file", {"path": {}, "dryRun": {}}))
        assert p["safeguard_params"] == ["dryRun"]

    def test_source_and_destination_count_as_targets(self):
        # move_file takes source/destination, not `path`; missing them would report
        # an irreversible tool as having no target at all.
        p = profile_tool(tool("move_file", {"source": {}, "destination": {}}))
        assert p["target_params"] == ["destination", "source"]
        assert p["irreversible"]

    def test_plural_target_param_is_found(self):
        p = profile_tool(tool("read_multiple_files", {"paths": {}}))
        assert p["target_params"] == ["paths"]

    def test_free_form_input_is_reported_with_the_parameter(self):
        p = profile_tool(tool("search_nodes", {"query": {}}))
        assert p["takes_free_form_input"] and p["free_form_params"] == ["query"]

    def test_tool_without_free_form_input_is_not_flagged(self):
        p = profile_tool(tool("list_directory", {"path": {}}))
        assert not p["takes_free_form_input"]

    def test_required_params_are_recorded(self):
        p = profile_tool(tool("write_file", {"path": {}, "content": {}}, required=["path"]))
        assert p["required_params"] == ["path"]

    def test_missing_schema_does_not_crash(self):
        p = profile_tool({"name": "x"})
        assert p["param_count"] == 0 and p["target_params"] == []

    def test_every_classification_carries_its_basis(self):
        p = profile_tool(tool("delete_thing"))
        assert p["effect_basis"] and p["irreversible_basis"]


class TestProfile:
    def test_against_a_real_server(self):
        p = profile(SERVER, "fake-store")
        assert p["schema"] == PROFILE_SCHEMA
        assert p["counts"] == {"tools": 2, "mutating": 1, "irreversible": 0,
                              "irreversible_without_a_confirmation_parameter": 0,
                              "accepting_free_form_input": 0}
        assert p["server"] == "fake-store"
        assert p["server_info"]["serverInfo"]["name"] == "fake-store"

    def test_no_tool_is_actually_called(self):
        """The profile reads the declared surface; calling tools would be a side effect."""
        from ajantik.mcp import MCPServer

        profile(SERVER)
        with MCPServer(SERVER) as server:   # a fresh process: store must be untouched
            assert json.loads(server.call_tool("read_all", {})["content"][0]["text"]) == {}

    def test_output_carries_no_score_and_no_grade(self):
        p = profile(SERVER)
        assert "no_score" in p          # the refusal is stated explicitly
        # Scan everything except that field, whose text necessarily names what it refuses.
        rest = {k: v for k, v in p.items() if k != "no_score"}
        blob = json.dumps(rest, ensure_ascii=False).lower()
        for forbidden in ("score", "grade", "rating", "/100"):
            assert forbidden not in blob, forbidden

    def test_limits_are_part_of_the_output_not_a_footnote(self):
        p = profile(SERVER)
        assert p["measured"] and len(p["not_measured"]) >= 3
        assert any("Behaviour" in item for item in p["not_measured"])

    def test_unreachable_server_raises_rather_than_returning_an_empty_profile(self):
        from ajantik.mcp import MCPError

        with pytest.raises((MCPError, OSError)):
            profile([sys.executable, "-c", "raise SystemExit(1)"])


class TestRendering:
    def test_markdown_has_counts_limits_and_no_score(self):
        text = render_markdown(profile(SERVER, "fake-store"))
        assert "Declared tools" in text and "Not measured" in text
        assert "No score" in text
        for forbidden in ("Score", "/100", "Grade"):
            assert forbidden not in text

    def test_index_entry_is_counts_only(self):
        entry = index_entry(profile(SERVER, "fake-store"))
        assert set(entry) == {"server", "counts"}

    def test_write_profile_writes_both_formats(self, tmp_path):
        p = profile(SERVER, "fake-store")
        write_profile(p, str(tmp_path / "p.json"), str(tmp_path / "p.md"))
        assert json.loads((tmp_path / "p.json").read_text())["schema"] == PROFILE_SCHEMA
        assert "Tool surface" in (tmp_path / "p.md").read_text()
