"""Live sender: forwards new sanitized observer records to ajantik.ai (stdlib only).

Reads `<state-dir>/events.jsonl` written by `observer.py`, revalidates every line with
`observer.validate_record`, rebuilds only `observer.FIELDS` and posts batches to the fixed origin.
The ingest token lives only in `<state-dir>/live.json` (0600) and is never printed. Pairing
starts at the current end of the log: history written before `pair` is not uploaded. The cursor
moves only after the server acknowledges a batch, so a retry replays the same event ids.

    python -m ajantik.live_sender pair  --state-dir DIR   # pairing code read with getpass
    python -m ajantik.live_sender sync  --state-dir DIR
    python -m ajantik.live_sender watch --state-dir DIR   # every 3 s until Ctrl-C

Also runs as `python live_sender.py ...` when saved next to `observer.py`.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import ssl
import stat
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from . import observer
else:  # standalone copy next to observer.py
    import observer

ORIGIN = "https://ajantik.ai"
CLAIM_URL = ORIGIN + "/api/live/claim"
EVENTS_URL = ORIGIN + "/api/live/events"
LIVE_SCHEMA = "ajantik.live.state.v1"
TIMEOUT_S = 10
POLL_S = 3
MAX_EVENTS = 50
MAX_PAYLOAD = 64 * 1024
MAX_READ = 256 * 1024
MAX_RESPONSE = 16 * 1024
USER_AGENT = "ajantik-live-sender/1"

PAIR_RE = re.compile(r"pair_[0-9a-f]{64}")
TOKEN_RE = re.compile(r"ingest_[0-9a-f]{64}")
CONNECTION_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
EXPIRES_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{1,6})?(Z|[+-]\d\d:\d\d)")


class LiveError(Exception):
    """Error whose message is safe to print (never a token, code, payload or response body).

    `fatal` errors stop `watch`; others (network, timeout, 5xx, 429) are retried next poll.
    """

    def __init__(self, message: str, fatal: bool = True):
        super().__init__(message)
        self.fatal = fatal


# --- HTTP ------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect so the Authorization header is never forwarded anywhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LiveError(f"the server asked for a redirect (HTTP {code}); refused")


def build_opener(*extra_handlers) -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()  # certificate and hostname verification on
    return urllib.request.build_opener(
        _NoRedirect(), urllib.request.HTTPSHandler(context=ctx), *extra_handlers
    )


_STATUS_TEXT = {
    401: "authorisation refused (HTTP 401); pair again",
    403: "authorisation refused (HTTP 403); pair again",
    409: "the server reported a conflict (HTTP 409)",
    413: "the request was too large (HTTP 413)",
}


def _post(opener, url: str, body: dict, token: str | None = None) -> dict:
    """POST JSON and return the JSON object reply. Errors never include bodies or secrets."""
    data = json.dumps(body, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": USER_AGENT}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with opener.open(req, timeout=TIMEOUT_S) as resp:
            if resp.geturl() != url:
                raise LiveError("the reply came from an unexpected address; refused")
            raw = resp.read(MAX_RESPONSE + 1)
    except LiveError:
        raise
    except urllib.error.HTTPError as e:
        e.close()
        code = e.code
        if 300 <= code < 400:
            raise LiveError(f"the server asked for a redirect (HTTP {code}); refused") from None
        if code in _STATUS_TEXT:
            raise LiveError(_STATUS_TEXT[code]) from None
        transient = code == 429 or code >= 500
        raise LiveError(f"server error (HTTP {code})", fatal=not transient) from None
    except TimeoutError:
        raise LiveError("the server timed out", fatal=False) from None
    except ssl.SSLError:
        raise LiveError("TLS verification failed", fatal=False) from None
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLError):
            raise LiveError("TLS verification failed", fatal=False) from None
        raise LiveError("network error; could not reach the server", fatal=False) from None
    except OSError:
        raise LiveError("network error; could not reach the server", fatal=False) from None
    if len(raw) > MAX_RESPONSE:
        raise LiveError("the server reply is too large; refused", fatal=False)
    try:
        reply = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise LiveError("the server reply is not valid JSON", fatal=False) from None
    if not isinstance(reply, dict):
        raise LiveError("the server reply is not in the expected format", fatal=False)
    return reply


# --- local state -----------------------------------------------------------------------------

def _live_path(state_dir: Path) -> Path:
    return state_dir / "live.json"


@contextmanager
def _sender_lock(state_dir: Path):
    """Exclusive, non-blocking lock so two senders never share a cursor."""
    try:
        fd = os.open(state_dir / ".live.lock", os.O_RDWR | os.O_CREAT, 0o600)
    except FileNotFoundError:
        raise LiveError("no state directory; run observer `init` first") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LiveError("another sender is running (lock held)") from None
        yield
    finally:
        os.close(fd)


def _parse_expires(value: object) -> datetime | None:
    if not isinstance(value, str) or not EXPIRES_RE.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _valid_cursor(c: object) -> bool:
    def nonneg_int(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0

    if not isinstance(c, dict) or set(c) != {"offset", "dev", "ino"}:
        return False
    if not nonneg_int(c["offset"]):
        return False
    if c["ino"] is None:
        return c["dev"] is None and c["offset"] == 0
    return nonneg_int(c["dev"]) and nonneg_int(c["ino"])


def load_live(state_dir: Path) -> dict:
    path = _live_path(state_dir)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        raise LiveError("no connection; run `pair` first") from None
    except OSError:
        raise LiveError("could not read live.json") from None
    try:
        if stat.S_IMODE(os.fstat(fd).st_mode) & 0o077:
            raise LiveError("live.json is readable by others; set its mode to 0600")
        with os.fdopen(fd, encoding="utf-8", closefd=False) as f:
            live = json.loads(f.read(MAX_RESPONSE))
    except (OSError, ValueError):
        raise LiveError("could not read live.json") from None
    finally:
        os.close(fd)
    if not (
        isinstance(live, dict)
        and live.get("schema") == LIVE_SCHEMA
        and live.get("origin") == ORIGIN
        and isinstance(live.get("connection_id"), str)
        and CONNECTION_RE.fullmatch(live["connection_id"])
        and isinstance(live.get("ingest_token"), str)
        and TOKEN_RE.fullmatch(live["ingest_token"])
        and _parse_expires(live.get("expires_at")) is not None
        and isinstance(live.get("operator_id"), str)
        and isinstance(live.get("agent_id"), str)
        and _valid_cursor(live.get("cursor"))
    ):
        raise LiveError("live.json is invalid; pair again")
    return live


def _save_live(state_dir: Path, live: dict) -> None:
    observer._write_private(_live_path(state_dir), json.dumps(live, indent=2) + "\n")


# --- pair ------------------------------------------------------------------------------------

def _eof_cursor(state_dir: Path) -> dict:
    """Cursor just after the last complete line of events.jsonl (offset 0 if absent)."""
    events = observer._paths(state_dir)["events"]
    with observer._locked(state_dir):
        try:
            fd = os.open(events, os.O_RDONLY)
        except FileNotFoundError:
            return {"offset": 0, "dev": None, "ino": None}
        try:
            st = os.fstat(fd)
            pos = st.st_size
            # Walk back to the last newline so a half-written line is sent once it completes.
            while pos > 0:
                start = max(0, pos - 64 * 1024)
                chunk = os.pread(fd, pos - start, start)
                i = chunk.rfind(b"\n")
                if i >= 0:
                    pos = start + i + 1
                    break
                pos = start
        finally:
            os.close(fd)
    return {"offset": pos, "dev": st.st_dev, "ino": st.st_ino}


def pair(state_dir: Path, code: str, opener=None) -> dict:
    """Claim a pairing code; store the token in live.json (0600). Returns public info only."""
    if not isinstance(code, str) or not PAIR_RE.fullmatch(code):
        raise LiveError("invalid pairing code format (pair_ + 64 lower-case hex digits)")
    try:
        state = observer._read_state(state_dir)
    except observer.ObserverError as e:
        raise LiveError(str(e)) from None
    opener = opener or build_opener()
    with _sender_lock(state_dir):
        reply = _post(opener, CLAIM_URL, {
            "pairing_code": code,
            "operator_id": state["operator_id"],
            "agent_id": state["agent_id"],
        })
        token, conn = reply.get("ingest_token"), reply.get("connection_id")
        expires = reply.get("expires_at")
        if not (isinstance(token, str) and TOKEN_RE.fullmatch(token)
                and isinstance(conn, str) and CONNECTION_RE.fullmatch(conn)
                and _parse_expires(expires) is not None):
            raise LiveError("the server reply is not in the expected format")
        live = {
            "schema": LIVE_SCHEMA,
            "origin": ORIGIN,
            "connection_id": conn,
            "ingest_token": token,
            "expires_at": expires,
            "operator_id": state["operator_id"],
            "agent_id": state["agent_id"],
            "cursor": _eof_cursor(state_dir),
        }
        _save_live(state_dir, live)
    return {"expires_at": expires}


# --- sync ------------------------------------------------------------------------------------

def _read_chunk(state_dir: Path, live: dict) -> bytes | None:
    """Bytes from the cursor (at most MAX_READ); None if there is no log yet.

    Binds the cursor to the file on first sight; refuses a replaced or truncated file.
    """
    events = observer._paths(state_dir)["events"]
    cur = live["cursor"]
    with observer._locked(state_dir):
        try:
            fd = os.open(events, os.O_RDONLY)
        except FileNotFoundError:
            if cur["ino"] is None:
                return None
            raise LiveError("events.jsonl is gone or was moved; sending stopped "
                            "(pair again)") from None
        try:
            st = os.fstat(fd)
            if cur["ino"] is None:
                cur.update(offset=0, dev=st.st_dev, ino=st.st_ino)
                _save_live(state_dir, live)
            elif (st.st_dev, st.st_ino) != (cur["dev"], cur["ino"]):
                raise LiveError("events.jsonl was replaced (a different file); sending stopped "
                                "(pair again)")
            if st.st_size < cur["offset"]:
                raise LiveError("events.jsonl was truncated; sending stopped (pair again)")
            return os.pread(fd, MAX_READ, cur["offset"])
        finally:
            os.close(fd)


def _next_batch(state_dir: Path, chunk: bytes, offset: int, state: dict):
    """Split `chunk` into (events, consumed_bytes, invalid_count).

    Stops before a partial last line and before the line that would break a request limit.
    A complete line longer than MAX_READ is skipped as invalid.
    """
    events_out: list[dict] = []
    seen: set[str] = set()
    payload_len = len(b'{"events":[]}')
    consumed = invalid = 0
    if chunk and b"\n" not in chunk and len(chunk) == MAX_READ:
        return [], _skip_long_line(state_dir, offset), 1
    while True:
        nl = chunk.find(b"\n", consumed)
        if nl < 0:
            break  # partial line: wait for the observer to finish it
        line = chunk[consumed:nl]
        end = nl + 1
        if not line.strip():
            consumed = end
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            rec = None
        if not observer.validate_record(rec, state) or rec["event_id"] in seen:
            invalid += 1
            consumed = end
            continue
        clean = {k: rec[k] for k in observer.FIELDS}
        enc_len = len(json.dumps(clean, separators=(",", ":")).encode())
        extra = enc_len + (1 if events_out else 0)
        if len(events_out) >= MAX_EVENTS or payload_len + extra > MAX_PAYLOAD:
            if not events_out:  # cannot fit even alone
                invalid += 1
                consumed = end
                continue
            break
        events_out.append(clean)
        seen.add(rec["event_id"])
        payload_len += extra
        consumed = end
    return events_out, consumed, invalid


def _skip_long_line(state_dir: Path, offset: int) -> int:
    """Bytes up to and including the next newline after `offset`; 0 if none yet."""
    events = observer._paths(state_dir)["events"]
    with open(events, "rb") as f:
        f.seek(offset)
        skipped = 0
        while block := f.read(MAX_READ):
            i = block.find(b"\n")
            if i >= 0:
                return skipped + i + 1
            skipped += len(block)
    return 0


def sync(state_dir: Path, opener=None, now: datetime | None = None) -> tuple[int, int]:
    """Send all complete new lines. Returns (sent, invalid_skipped)."""
    try:
        state = observer._read_state(state_dir)
    except observer.ObserverError as e:
        raise LiveError(str(e)) from None
    opener = opener or build_opener()
    sent = invalid_total = 0
    with _sender_lock(state_dir):
        live = load_live(state_dir)
        if (live["operator_id"], live["agent_id"]) != (state["operator_id"], state["agent_id"]):
            raise LiveError("the observer identity changed after pairing; pair again")
        if (now or datetime.now(UTC)) >= _parse_expires(live["expires_at"]):
            raise LiveError("the connection has expired; pair again")
        while True:
            chunk = _read_chunk(state_dir, live)
            if chunk is None:
                break
            offset = live["cursor"]["offset"]
            batch, consumed, invalid = _next_batch(state_dir, chunk, offset, state)
            if consumed == 0:
                break
            if batch:
                reply = _post(opener, EVENTS_URL, {"events": batch}, live["ingest_token"])
                if reply.get("accepted") is not True:
                    raise LiveError("the server did not accept the batch; cursor not advanced",
                                    fatal=False)
            # Advance only after the ack (or when the range held nothing sendable).
            live["cursor"]["offset"] = offset + consumed
            _save_live(state_dir, live)
            sent += len(batch)
            invalid_total += invalid
    return sent, invalid_total


def watch(state_dir: Path, opener=None, sleep=time.sleep, iterations: int | None = None) -> int:
    """Poll `sync` every POLL_S seconds. Stops on Ctrl-C or a fatal error."""
    last_error = None
    n = 0
    try:
        while iterations is None or n < iterations:
            n += 1
            try:
                sent, invalid = sync(state_dir, opener)
                last_error = None
                _report(sent, invalid, quiet=True)
            except LiveError as e:
                if e.fatal:
                    _warn(str(e))
                    return 2
                if str(e) != last_error:  # do not repeat the same transient warning each poll
                    _warn(f"{e}; retrying in {POLL_S} s")
                last_error = str(e)
            sleep(POLL_S)
    except KeyboardInterrupt:
        print("ajantik live: stopped", file=sys.stderr)
    return 0


# --- CLI -------------------------------------------------------------------------------------

def _warn(msg: str) -> None:
    print(f"ajantik live: warning: {msg}", file=sys.stderr)


def _report(sent: int, invalid: int, quiet: bool = False) -> None:
    if invalid:
        _warn(f"{invalid} invalid or duplicate lines skipped (content not shown)")
    if sent or not quiet:
        print(f"ajantik live: {sent} events sent", file=sys.stderr)


def main(argv: list[str] | None = None, opener=None, prompt=getpass.getpass) -> int:
    parser = argparse.ArgumentParser(prog="python -m ajantik.live_sender")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("pair", "read the pairing code at a hidden prompt and save the connection"),
        ("sync", "send new records once"),
        ("watch", f"send every {POLL_S} s (stop with Ctrl-C)"),
    ):
        sub.add_parser(name, help=text).add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    state_dir = args.state_dir.expanduser()
    try:
        if args.command == "pair":
            try:
                code = prompt("Pairing code (hidden): ").strip()
            except (EOFError, KeyboardInterrupt):
                _warn("pairing cancelled")
                return 2
            info = pair(state_dir, code, opener)
            print("ajantik live: paired; only records written from now on will be sent")
            print(f"expires_at: {info['expires_at']}")
            return 0
        if args.command == "sync":
            _report(*sync(state_dir, opener))
            return 0
        return watch(state_dir, opener)
    except LiveError as e:
        _warn(str(e))
        return 2
    except Exception as e:  # noqa: BLE001 - never print request bodies, tokens or codes
        _warn(f"unexpected error ({type(e).__name__})")
        return 2


if __name__ == "__main__":
    sys.exit(main())
