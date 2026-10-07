import io
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from ajantik import observer as ob

CANARY = "CANARY-7f3a-leak"


def run_cli(*args, stdin=b""):
    return subprocess.run(
        [sys.executable, "-m", "ajantik.observer", *map(str, args)],
        input=stdin, capture_output=True, timeout=30, check=False,
    )


def payload(event="PreToolUse", session="sess-1", **extra):
    base = {"hook_event_name": event, "session_id": session}
    base.update(extra)
    return json.dumps(base).encode()


def events(d):
    return [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]


@pytest.fixture
def state(tmp_path):
    d = tmp_path / "obs"
    ob.init_state(d, "cfg_a")
    return d


def test_init_permissions_and_output_hide_secret(tmp_path):
    d = tmp_path / "obs"
    r = run_cli("init", "--state-dir", d, "--configuration-id", "cfg_a")
    assert r.returncode == 0
    key = (d / "secret.key").read_text().strip()
    assert key not in r.stdout.decode() + r.stderr.decode()
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    for name in ("state.json", "secret.key", ".lock"):
        assert stat.S_IMODE(os.stat(d / name).st_mode) == 0o600
    ob.collect(d, payload())
    assert stat.S_IMODE(os.stat(d / "events.jsonl").st_mode) == 0o600


def test_reinit_keeps_ids_and_secret_but_changes_config(state):
    first = json.loads((state / "state.json").read_text())
    key = (state / "secret.key").read_text()
    ob.collect(state, payload(session="s"))
    second = ob.init_state(state, "cfg_b")
    assert (second["operator_id"], second["agent_id"]) == (first["operator_id"], first["agent_id"])
    assert (state / "secret.key").read_text() == key
    ob.collect(state, payload(session="s"))
    a, b = events(state)
    assert (a["configuration_id"], b["configuration_id"]) == ("cfg_a", "cfg_b")
    assert a["agent_id"] == b["agent_id"] and a["run_id"] == b["run_id"]
    assert b["sequence"] == 2


@pytest.mark.parametrize("label", ["", "../x", "a b", "-x", "a" * 65, "cfg\n", "\u00e9fg"])
def test_bad_configuration_label_rejected(tmp_path, label):
    with pytest.raises(ob.ObserverError):
        ob.init_state(tmp_path / "obs", label)


def test_canary_redaction(state):
    raws = [
        payload("UserPromptSubmit", session=CANARY, prompt=CANARY),  # unknown: ignored
        payload("SessionStart", session=CANARY, cwd=f"/home/{CANARY}", model=CANARY,
                transcript_path=f"/tmp/{CANARY}.jsonl", source=CANARY),
        payload("PreToolUse", session=CANARY, tool_name="Bash", tool_use_id=CANARY,
                tool_input={"command": f"echo {CANARY}"}, permission_mode=CANARY),
        payload("PostToolUse", session=CANARY, tool_name=f"mcp__{CANARY}__x", tool_use_id=CANARY,
                tool_response={"stdout": CANARY}),
        payload("PostToolUseFailure", session=CANARY, tool_name=CANARY, tool_use_id=CANARY,
                error=CANARY, is_interrupt=False),
        payload("Stop", session=CANARY, last_assistant_message=CANARY),
        payload("SessionEnd", session=CANARY, reason=CANARY),
    ]
    for raw in raws:
        r = run_cli("collect", "--state-dir", state, stdin=raw)
        assert r.returncode == 0 and r.stdout == b""
        assert CANARY.encode() not in r.stderr
    log = (state / "events.jsonl").read_text()
    assert CANARY not in log and "home" not in log
    recs = events(state)
    assert [r["event_type"] for r in recs] == [
        "session_start", "tool_requested", "tool_returned", "tool_failed", "turn_finished",
        "session_end",
    ]
    assert [r["tool"] for r in recs] == ["none", "Bash", "mcp", "other", "none", "none"]
    assert all(list(r) == list(ob.FIELDS) for r in recs)
    out = io.StringIO()
    assert ob.export(state, out) == (6, 0)
    assert CANARY not in out.getvalue()


def test_tool_normalization():
    assert ob.normalize_tool("Read") == "Read"
    assert ob.normalize_tool("mcp__github__create_issue") == "mcp"
    assert ob.normalize_tool("read") == "other"
    assert ob.normalize_tool("MultiEdit") == "other"
    assert ob.normalize_tool(None) == "other"
    assert ob.normalize_tool(["Bash"]) == "other"


def test_hmac_pseudonyms(state, tmp_path):
    ob.collect(state, payload(session="s1", tool_name="Read", tool_use_id="toolu_1"))
    ob.collect(state, payload("PostToolUse", session="s1", tool_name="Read", tool_use_id="toolu_1"))
    ob.collect(state, payload(session="toolu_1", tool_name="Read", tool_use_id="s1"))
    a, b, c = events(state)
    assert a["run_id"] == b["run_id"] and a["tool_call_id"] == b["tool_call_id"]
    assert a["run_id"].startswith("run_") and a["tool_call_id"].startswith("call_")
    # Separate domains: the same raw string never gives the same digest.
    assert a["run_id"][4:] != c["tool_call_id"][5:]
    assert a["tool_call_id"][5:] != c["run_id"][4:]
    # A different secret gives different pseudonyms for the same session.
    other = tmp_path / "other"
    ob.init_state(other, "cfg_a")
    ob.collect(other, payload(session="s1", tool_use_id="toolu_1"))
    (d,) = events(other)
    assert d["run_id"] != a["run_id"] and d["tool_call_id"] != a["tool_call_id"]
    ob.collect(state, payload(session="s1", tool_name="Read"))
    assert events(state)[-1]["tool_call_id"] is None


def test_sequence_per_run_across_processes(state):
    def one(i):
        return run_cli("collect", "--state-dir", state, stdin=payload(session=f"s{i % 2}"))

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, range(24)))
    assert all(r.returncode == 0 and r.stdout == b"" for r in results)
    by_run: dict[str, list[int]] = {}
    for r in events(state):
        by_run.setdefault(r["run_id"], []).append(r["sequence"])
    assert sorted(len(v) for v in by_run.values()) == [12, 12]
    for seqs in by_run.values():
        assert seqs == list(range(1, 13))  # appended under the lock, so file order == sequence


def test_sequence_survives_lost_counter(state):
    for _ in range(3):
        ob.collect(state, payload())
    (state / "sequences.json").unlink()
    ob.collect(state, payload())
    assert [r["sequence"] for r in events(state)] == [1, 2, 3, 4]


@pytest.mark.parametrize("raw", [
    b"", b"not json", b"[1,2]", b"\xff\xfe", b'"str"', b"{" * 10,
    json.dumps({"hook_event_name": "PreToolUse"}).encode(),  # missing session
    json.dumps({"hook_event_name": "PreToolUse", "session_id": 5}).encode(),
    json.dumps({"hook_event_name": "Notification", "session_id": "s"}).encode(),
    b'{"hook_event_name":"PreToolUse","session_id":"s","x":"' + b"a" * ob.MAX_INPUT_BYTES + b'"}',
], ids=lambda raw: f"{len(raw)}b")
def test_malformed_input_fails_open(state, raw):
    r = run_cli("collect", "--state-dir", state, stdin=raw)
    assert r.returncode == 0 and r.stdout == b""
    assert not (state / "events.jsonl").exists()


def test_collect_without_init_fails_open(tmp_path):
    r = run_cli("collect", "--state-dir", tmp_path / "none", stdin=payload())
    assert r.returncode == 0 and r.stdout == b"" and b"init" in r.stderr


def test_export_drops_tampered_records(state):
    ob.collect(state, payload(tool_name="Bash", tool_use_id="t1"))
    ob.collect(state, payload("Stop"))
    good = events(state)
    st = json.loads((state / "state.json").read_text())
    bad = [
        {**good[0], "prompt": CANARY},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000001", "tool": CANARY},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000002", "run_id": CANARY},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000003", "sequence": 0},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000004", "sequence": True},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000005", "event_type": ["x"]},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000006",
         "operator_id": "op_" + "0" * 32},
        {**good[0], "event_id": "0b5c7c4e-0000-4000-8000-000000000007",
         "occurred_at": f"2026-01-01T00:00:00.000Z{CANARY}"},
        {**good[1], "event_id": "0b5c7c4e-0000-4000-8000-000000000008", "tool": "Bash"},
        {**good[1], "event_id": "0b5c7c4e-0000-4000-8000-000000000009",
         "tool_call_id": good[0]["tool_call_id"]},
        {k: v for k, v in good[0].items() if k != "tool_call_id"},
        good[0],  # duplicate event_id
    ]
    with (state / "events.jsonl").open("a") as f:
        for rec in bad:
            f.write(json.dumps(rec) + "\n")
        f.write(f"{{broken {CANARY}\n")
    assert st["operator_id"] == good[0]["operator_id"]
    r = run_cli("export", "--state-dir", state)
    assert r.returncode == 0
    out = r.stdout.decode()
    assert CANARY not in out and CANARY.encode() not in r.stderr
    assert [json.loads(x) for x in out.splitlines()] == good
    assert b"13 " in r.stderr
