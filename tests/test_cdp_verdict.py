"""Verdicts from a guard's record: the record is the world, the report is the claim."""

from __future__ import annotations

import json
from pathlib import Path

from ajantik import cdp_verdict
from ajantik.cdp_verdict import Record

ADAPTER = '''
import re

def world(record, project):
    saved = {record.request_json(r)["name"] for r in record.writes("/items") if record.ok(r)}
    return {name: [] if name in saved else ["not saved"] for name in ("a", "b", "c")}

def claims(message, project):
    return {name: "done" if f"{name}: ok" in message else "not_done" for name in ("a", "b", "c")}

def damage(record, project):
    return {"c": ["written to another account"]} if record.writes("/other") else {}
'''


def _record(tmp: Path, rows: list[dict]) -> Path:
    p = tmp / "run.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _put(name: str, status: int, url: str = "https://x/items") -> dict:
    return {"kind": "http", "method": "PUT", "url": url, "status": status,
            "request": {"body": json.dumps({"name": name})}}


def test_record_answers_what_the_system_accepted(tmp_path):
    rec = Record.load(_record(tmp_path, [
        _put("a", 204), _put("b", 503),
        {"kind": "http", "method": "GET", "url": "https://x/items", "status": 200},
        {"kind": "blocked", "reason": "write to a denied URL", "method": "POST",
         "url": "https://x/submit"},
    ]))
    assert [r["status"] for r in rec.writes()] == [204, 503]
    assert [r["status"] for r in rec.failed_writes()] == [503]
    assert rec.request_json(rec.last_ok("/items")) == {"name": "a"}
    assert len(rec.blocked()) == 1


def test_judge_sets_the_report_against_the_record(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER)
    record = _record(tmp_path, [_put("a", 204), _put("b", 503), _put("c", 204),
                                _put("c", 204, "https://x/other")])
    message = tmp_path / "report.txt"
    message.write_text("a: ok\nb: ok\nc: failed\n")
    r = cdp_verdict.judge_files(adapter, record, message)
    v = {u: x["verdict"] for u, x in r["units"].items()}
    assert v == {"a": "correct", "b": "silent_wrong", "c": "over_cautious"}
    assert r["units"]["c"]["damage"] == ["written to another account"]
    assert len(r["failed_writes"]) == 1
    text = "\n".join(cdp_verdict.summary(r))
    assert "SILENT WRONG" in text and "1 refused by the system" in text


def test_latest_is_the_last_state_the_system_showed(tmp_path):
    get = {"kind": "http", "method": "GET", "url": "https://x/items/1", "status": 200,
           "response": {"body": json.dumps({"name": "old"})}}
    rec = Record.load(_record(tmp_path, [_put("new", 204, "https://x/items/1"), get]))
    assert rec.latest("/items/1") == {"name": "old"}       # a later read wins
    rec = Record.load(_record(tmp_path, [get, _put("new", 204, "https://x/items/1"),
                                         _put("lost", 401, "https://x/items/1")]))
    assert rec.latest("/items/1") == {"name": "new"}       # a refused write does not count


class FakeReviewer:
    name = "fake"

    def ask(self, prompt: str) -> str:
        said = "done" if "unit a" in prompt else "not_done"
        return json.dumps({"claim": said, "reason": "read the message"})


def test_a_reviewer_reads_a_free_text_report_per_unit(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER + '\nUNITS = {"a": "unit a", "b": "unit b", "c": "unit c"}\n')
    record = _record(tmp_path, [_put("a", 204), _put("b", 204), _put("c", 204)])
    message = tmp_path / "report.txt"
    message.write_text("I saved a. b and c failed.")
    r = cdp_verdict.judge_files(adapter, record, message, reviewer=FakeReviewer())
    assert {u: x["verdict"] for u, x in r["units"].items()} == {
        "a": "correct", "b": "over_cautious", "c": "over_cautious"}
    assert r["units"]["a"]["reason"] == "read the message" and r["reviewer"] == "fake"
