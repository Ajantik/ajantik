import email.message
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
import urllib.response
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ajantik import live_sender as ls
from ajantik import observer as ob

CODE = "pair_" + "a" * 64
TOKEN = "ingest_" + "b" * 64
EXPIRES = "2099-01-01T00:00:00Z"
CANARY = "CANARY-9c1e-leak"
SRC = Path(ls.__file__).parent


class FakeResp:
    def __init__(self, url, body):
        self.url, self.body = url, body

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Records requests; each reply is a dict (JSON 200), an int (HTTP error) or an exception."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def open(self, req, timeout=None):
        self.requests.append({
            "url": req.full_url, "timeout": timeout, "headers": dict(req.header_items()),
            "body": json.loads(req.data),
        })
        reply = self.replies.pop(0) if self.replies else {"accepted": True}
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            raise urllib.error.HTTPError(req.full_url, reply, "x", {},
                                         io.BytesIO(f"body {CANARY}".encode()))
        return FakeResp(req.full_url, json.dumps(reply).encode())

    def sent_ids(self):
        return [e["event_id"] for r in self.requests for e in r["body"].get("events", [])]


def claim_reply():
    return {"connection_id": "conn_1", "ingest_token": TOKEN, "expires_at": EXPIRES}


def hook(event="PreToolUse", session="s1", **extra):
    return json.dumps({"hook_event_name": event, "session_id": session,
                       "tool_name": "Bash", "tool_use_id": "tu", **extra}).encode()


def log_ids(d):
    return [json.loads(x)["event_id"] for x in (d / "events.jsonl").read_text().splitlines()]


def cursor(d):
    return json.loads((d / "live.json").read_text())["cursor"]


@pytest.fixture
def state(tmp_path):
    d = tmp_path / "obs"
    ob.init_state(d, "cfg_a")
    return d


@pytest.fixture
def paired(state):
    ls.pair(state, CODE, FakeOpener(claim_reply()))
    return state


def test_pair_stores_0600_and_never_prints_secrets(state, capsys):
    ob.collect(state, hook())
    opener = FakeOpener(claim_reply())
    assert ls.main(["pair", "--state-dir", str(state)], opener, prompt=lambda _: CODE) == 0
    out = capsys.readouterr()
    shown = out.out + out.err
    assert TOKEN not in shown and CODE not in shown and "conn_1" not in shown
    assert EXPIRES in shown
    assert stat.S_IMODE(os.stat(state / "live.json").st_mode) == 0o600
    live = json.loads((state / "live.json").read_text())
    assert live["ingest_token"] == TOKEN
    ids = json.loads((state / "state.json").read_text())
    (req,) = opener.requests
    assert req["url"] == "https://ajantik.ai/api/live/claim" and req["timeout"] == 10
    assert req["body"] == {"pairing_code": CODE, "operator_id": ids["operator_id"],
                           "agent_id": ids["agent_id"]}
    assert "Authorization" not in req["headers"]


def test_pair_rejects_bad_code_and_bad_reply_without_saving(state):
    opener = FakeOpener()
    with pytest.raises(ls.LiveError):
        ls.pair(state, "pair_" + "A" * 64, opener)
    assert opener.requests == []
    with pytest.raises(ls.LiveError):
        ls.pair(state, CODE, FakeOpener({**claim_reply(), "ingest_token": "short"}))
    assert not (state / "live.json").exists()


def test_pair_skips_prior_history_by_default(state):
    ob.collect(state, hook())
    ob.collect(state, hook(event="PostToolUse"))
    ls.pair(state, CODE, FakeOpener(claim_reply()))
    ob.collect(state, hook(event="Stop"))
    opener = FakeOpener()
    assert ls.sync(state, opener) == (1, 0)
    assert opener.sent_ids() == log_ids(state)[2:]
    assert ls.sync(state, opener) == (0, 0)
    assert len(opener.requests) == 1


def test_absent_log_binds_on_first_sync_from_offset_zero(paired):
    assert cursor(paired) == {"offset": 0, "dev": None, "ino": None}
    assert ls.sync(paired, FakeOpener()) == (0, 0)
    ob.collect(paired, hook())
    ob.collect(paired, hook(event="Stop"))
    opener = FakeOpener()
    assert ls.sync(paired, opener) == (2, 0)
    assert opener.sent_ids() == log_ids(paired)
    st = os.stat(paired / "events.jsonl")
    assert cursor(paired) == {"offset": st.st_size, "dev": st.st_dev, "ino": st.st_ino}


def test_exact_allowlist_and_invalid_lines_skipped_without_content(paired, capsys):
    ob.collect(paired, hook())
    good = json.loads((paired / "events.jsonl").read_text())
    extra = {**good, "event_id": "11111111-1111-4111-8111-111111111111", "prompt": CANARY}
    foreign = {**good, "event_id": "22222222-2222-4222-8222-222222222222",
               "operator_id": "op_" + "0" * 32}
    with open(paired / "events.jsonl", "a") as f:
        f.writelines(line + "\n" for line in (
            json.dumps(extra), json.dumps(foreign), f"not json {CANARY}", "",
            json.dumps(good),  # duplicate event_id
        ))
    opener = FakeOpener()
    assert ls.main(["sync", "--state-dir", str(paired)], opener) == 0
    (req,) = opener.requests
    (event,) = req["body"]["events"]
    assert list(event) == list(ob.FIELDS)
    assert event == {k: good[k] for k in ob.FIELDS}
    assert CANARY not in json.dumps(req["body"])
    out = capsys.readouterr()
    assert "4 invalid" in out.err and CANARY not in out.out + out.err
    assert req["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert req["url"] == "https://ajantik.ai/api/live/events" and req["timeout"] == 10
    assert cursor(paired)["offset"] == os.path.getsize(paired / "events.jsonl")


@pytest.mark.parametrize("failure", [500, 401, 403, 409, 413, TimeoutError(),
                                     urllib.error.URLError("x"), {"accepted": False}])
def test_failure_preserves_cursor_and_retry_replays_same_ids(paired, failure):
    ob.collect(paired, hook())
    ob.collect(paired, hook(event="Stop"))
    ls.sync(paired, FakeOpener())  # binds the file
    ob.collect(paired, hook(event="SessionEnd"))
    before = cursor(paired)
    failing = FakeOpener(failure)
    with pytest.raises(ls.LiveError):
        ls.sync(paired, failing)
    assert cursor(paired) == before
    retry = FakeOpener()
    assert ls.sync(paired, retry) == (1, 0)
    assert retry.sent_ids() == failing.sent_ids() == log_ids(paired)[2:]


def test_http_error_output_is_short_and_safe(paired, capsys):
    ob.collect(paired, hook())
    assert ls.main(["sync", "--state-dir", str(paired)], FakeOpener(401)) == 2
    out = capsys.readouterr()
    assert "401" in out.err and CANARY not in out.err + out.out and TOKEN not in out.err


def test_success_advances_and_batches_keep_order_and_limits(paired):
    for i in range(120):
        ob.collect(paired, hook(session=f"s{i % 3}"))
    opener = FakeOpener()
    assert ls.sync(paired, opener) == (120, 0)
    sizes = [len(r["body"]["events"]) for r in opener.requests]
    assert sizes == [50, 50, 20]
    assert opener.sent_ids() == log_ids(paired)
    for r in opener.requests:
        assert len(json.dumps(r["body"], separators=(",", ":")).encode()) <= ls.MAX_PAYLOAD


def test_batch_failure_midway_keeps_acknowledged_part(paired):
    for _ in range(60):
        ob.collect(paired, hook())
    opener = FakeOpener({"accepted": True}, 500)
    with pytest.raises(ls.LiveError):
        ls.sync(paired, opener)
    lines = (paired / "events.jsonl").read_bytes().splitlines(keepends=True)
    assert cursor(paired)["offset"] == sum(len(x) for x in lines[:50])
    retry = FakeOpener()
    ls.sync(paired, retry)
    assert retry.sent_ids() == log_ids(paired)[50:]


def test_partial_last_line_waits_without_consuming(paired):
    ob.collect(paired, hook())
    ls.sync(paired, FakeOpener())
    full = (paired / "events.jsonl").read_bytes()
    ob.collect(paired, hook(event="Stop"))
    data = (paired / "events.jsonl").read_bytes()
    new_line = data[len(full):]
    with open(paired / "events.jsonl", "r+b") as f:
        f.truncate(len(full))
    with open(paired / "events.jsonl", "ab") as f:
        f.write(new_line[:30])
    opener = FakeOpener()
    assert ls.sync(paired, opener) == (0, 0)
    assert opener.requests == [] and cursor(paired)["offset"] == len(full)
    with open(paired / "events.jsonl", "ab") as f:
        f.write(new_line[30:])
    assert ls.sync(paired, opener) == (1, 0)
    assert cursor(paired)["offset"] == len(data)


def test_pair_cursor_stops_before_partial_line(state):
    ob.collect(state, hook())
    size = os.path.getsize(state / "events.jsonl")
    with open(state / "events.jsonl", "ab") as f:
        f.write(b'{"half')
    ls.pair(state, CODE, FakeOpener(claim_reply()))
    assert cursor(state)["offset"] == size


def test_oversized_line_is_skipped_not_stuck(paired):
    ob.collect(paired, hook())
    ls.sync(paired, FakeOpener())
    with open(paired / "events.jsonl", "a") as f:
        f.write("x" * (ls.MAX_READ + 10) + "\n")
    ob.collect(paired, hook(event="Stop"))
    opener = FakeOpener()
    assert ls.sync(paired, opener) == (1, 1)
    assert opener.sent_ids() == [log_ids_last(paired)]


def log_ids_last(d):
    return json.loads((d / "events.jsonl").read_text().splitlines()[-1])["event_id"]


def test_rotation_truncation_and_deletion_refused(paired):
    ob.collect(paired, hook())
    ob.collect(paired, hook(event="Stop"))
    ls.sync(paired, FakeOpener())
    events = paired / "events.jsonl"
    before = cursor(paired)

    with open(events, "r+b") as f:  # truncation
        f.truncate(10)
    with pytest.raises(ls.LiveError, match="truncated"):
        ls.sync(paired, FakeOpener())
    assert cursor(paired) == before

    replacement = paired / "new.jsonl"  # rotation: same name, different file
    replacement.write_bytes(b"\n" * (before["offset"] + 5))
    os.replace(replacement, events)
    opener = FakeOpener()
    with pytest.raises(ls.LiveError, match="different file"):
        ls.sync(paired, opener)
    assert opener.requests == [] and cursor(paired) == before

    events.unlink()
    with pytest.raises(ls.LiveError, match="is gone"):
        ls.sync(paired, FakeOpener())


class _FakeHTTPS(urllib.request.HTTPSHandler):
    handler_order = 100

    def __init__(self):
        super().__init__()
        self.urls = []

    def https_open(self, req):
        self.urls.append(req.full_url)
        headers = email.message.Message()
        headers["Location"] = "https://evil.example/steal"
        resp = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, 302)
        resp.msg = "Found"
        return resp


def test_redirects_rejected_by_real_opener(paired):
    ob.collect(paired, hook())
    fake = _FakeHTTPS()
    opener = ls.build_opener(fake)
    with pytest.raises(ls.LiveError, match="redirect"):
        ls.sync(paired, opener)
    assert fake.urls == ["https://ajantik.ai/api/live/events"]  # never followed
    assert cursor(paired)["offset"] == 0
    with pytest.raises(ls.LiveError, match="redirect"):
        ls.sync(paired, FakeOpener(307))


def test_real_opener_verifies_tls():
    opener = ls.build_opener()
    (https,) = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
    ctx = https._context
    assert ctx.check_hostname and ctx.verify_mode.name == "CERT_REQUIRED"
    redirects = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
    assert redirects and all(isinstance(h, ls._NoRedirect) for h in redirects)


def test_expired_connection_and_identity_change_refused(paired):
    ob.collect(paired, hook())
    opener = FakeOpener()
    with pytest.raises(ls.LiveError, match="expired"):
        ls.sync(paired, opener, now=datetime(2100, 1, 1, tzinfo=UTC))
    assert opener.requests == []
    s = json.loads((paired / "state.json").read_text())
    s["agent_id"] = "agt_" + "f" * 32
    (paired / "state.json").write_text(json.dumps(s))
    with pytest.raises(ls.LiveError, match="identity"):
        ls.sync(paired, opener)


def test_lock_blocks_concurrent_sender(paired):
    with ls._sender_lock(paired), pytest.raises(ls.LiveError, match="lock held"):
        ls.sync(paired, FakeOpener())


def test_live_json_permissions_enforced(paired):
    os.chmod(paired / "live.json", 0o644)
    with pytest.raises(ls.LiveError, match="0600"):
        ls.sync(paired, FakeOpener())


def test_watch_retries_transient_stops_on_fatal_and_ctrl_c(paired):
    ob.collect(paired, hook())
    opener = FakeOpener(500, {"accepted": True})
    assert ls.watch(paired, opener, sleep=lambda _: None, iterations=2) == 0
    assert opener.sent_ids()[0] == opener.sent_ids()[1]
    ob.collect(paired, hook(event="Stop"))
    assert ls.watch(paired, FakeOpener(401), sleep=lambda _: None) == 2

    def interrupt(_):
        raise KeyboardInterrupt

    assert ls.watch(paired, FakeOpener(), sleep=interrupt) == 0


def test_pair_code_never_accepted_as_argument(state):
    with pytest.raises(SystemExit):
        ls.main(["pair", "--state-dir", str(state), CODE], FakeOpener())


@pytest.mark.parametrize("mode", ["package", "standalone"])
def test_invocation_modes(tmp_path, state, mode):
    if mode == "package":
        cmd = [sys.executable, "-m", "ajantik.live_sender"]
    else:
        for name in ("observer.py", "live_sender.py"):
            shutil.copy(SRC / name, tmp_path / name)
        cmd = [sys.executable, str(tmp_path / "live_sender.py")]
    r = subprocess.run([*cmd, "sync", "--state-dir", str(state)], capture_output=True,
                       timeout=30, check=False, cwd=tmp_path, stdin=subprocess.DEVNULL)
    assert r.returncode == 2
    assert "pair" in r.stderr.decode()  # "no connection; run `pair` first"
