"""Local observer for Claude Code hooks: sanitized, pseudonymous event log (stdlib only).

Writes `ajantik.observation.v1` records to `<state-dir>/events.jsonl`. Only allowlisted fields
are ever produced; prompts, paths, tool input/output, session ids, model names and error text
are never copied. The observer never blocks an action and proves nothing about completion,
correctness or completeness: it is a local observation, and the configuration label is declared
by the operator, not measured.

    python -m ajantik.observer init --state-dir DIR --configuration-id LABEL
    python -m ajantik.observer collect --state-dir DIR   < hook payload on stdin
    python -m ajantik.observer export --state-dir DIR    > sanitized.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "ajantik.observation.v1"
STATE_SCHEMA = "ajantik.observer.state.v1"
SOURCE = "claude-code-hook"
EVIDENCE_LEVEL = "local_observation"
CONFIGURATION_BASIS = "operator_declared"
MAX_INPUT_BYTES = 1024 * 1024

FIELDS = (
    "schema", "event_id", "operator_id", "agent_id", "configuration_id", "run_id", "sequence",
    "occurred_at", "source", "event_type", "tool", "evidence_level", "configuration_basis",
    "tool_call_id",
)
HOOK_EVENTS = {
    "SessionStart": "session_start",
    "PreToolUse": "tool_requested",
    "PostToolUse": "tool_returned",
    "PostToolUseFailure": "tool_failed",
    "Stop": "turn_finished",
    "SessionEnd": "session_end",
}
EVENT_TYPES = frozenset(HOOK_EVENTS.values())
TOOL_EVENTS = frozenset({"tool_requested", "tool_returned", "tool_failed"})
BUILTIN_TOOLS = frozenset({
    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "Agent", "Task",
    "NotebookEdit", "TodoWrite",
})
TOOLS = BUILTIN_TOOLS | {"mcp", "other", "none"}

CONFIG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
OPERATOR_RE = re.compile(r"op_[0-9a-f]{32}")
AGENT_RE = re.compile(r"agt_[0-9a-f]{32}")
RUN_RE = re.compile(r"run_[0-9a-f]{32}")
CALL_RE = re.compile(r"call_[0-9a-f]{32}")
TIME_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")

# Separate HMAC domains so a run_id can never equal a tool_call_id for the same raw string.
_RUN_DOMAIN = b"ajantik.observer.run_id.v1\x00"
_CALL_DOMAIN = b"ajantik.observer.tool_call_id.v1\x00"


class ObserverError(Exception):
    """Error whose message is safe to print (never contains payload content)."""


# --- state -----------------------------------------------------------------------------------

def _paths(state_dir: Path) -> dict[str, Path]:
    return {
        "state": state_dir / "state.json",
        "secret": state_dir / "secret.key",
        "events": state_dir / "events.jsonl",
        "sequences": state_dir / "sequences.json",
        "lock": state_dir / ".lock",
    }


@contextmanager
def _locked(state_dir: Path):
    fd = os.open(_paths(state_dir)["lock"], os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_private(path: Path, data: str) -> None:
    """Atomically replace `path` with a 0600 file."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, data.encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _check_config(label: str) -> str:
    if not isinstance(label, str) or not CONFIG_RE.fullmatch(label):
        raise ObserverError(
            "configuration-id: must be 1-64 characters of [A-Za-z0-9._-], starting with a letter or digit"
        )
    return label


def init_state(state_dir: Path, configuration_id: str) -> dict:
    """Create or update the state. Re-init keeps operator_id, agent_id and the secret."""
    _check_config(configuration_id)
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    p = _paths(state_dir)
    with _locked(state_dir):
        os.chmod(p["lock"], 0o600)
        state = _read_state(state_dir) if p["state"].exists() else None
        if state is None:
            state = {"operator_id": f"op_{uuid.uuid4().hex}", "agent_id": f"agt_{uuid.uuid4().hex}"}
        if not p["secret"].exists():
            _write_private(p["secret"], secrets.token_hex(32) + "\n")
        state = {
            "schema": STATE_SCHEMA,
            "operator_id": state["operator_id"],
            "agent_id": state["agent_id"],
            "configuration_id": configuration_id,
            "configuration_basis": CONFIGURATION_BASIS,
        }
        _write_private(p["state"], json.dumps(state, indent=2) + "\n")
        for name in ("secret", "state"):
            os.chmod(p[name], 0o600)
        if p["events"].exists():
            os.chmod(p["events"], 0o600)
    return state


def _read_state(state_dir: Path) -> dict:
    p = _paths(state_dir)
    try:
        state = json.loads(p["state"].read_text())
    except FileNotFoundError:
        raise ObserverError("no state; run `init` first") from None
    except (OSError, ValueError):
        raise ObserverError("could not read the state file") from None
    if not (
        isinstance(state, dict)
        and isinstance(state.get("operator_id"), str) and OPERATOR_RE.fullmatch(state["operator_id"])
        and isinstance(state.get("agent_id"), str) and AGENT_RE.fullmatch(state["agent_id"])
        and isinstance(state.get("configuration_id"), str)
        and CONFIG_RE.fullmatch(state["configuration_id"])
    ):
        raise ObserverError("the state file is invalid")
    return state


def _read_secret(state_dir: Path) -> bytes:
    try:
        key = bytes.fromhex(_paths(state_dir)["secret"].read_text().strip())
    except FileNotFoundError:
        raise ObserverError("no secret key; run `init` first") from None
    except (OSError, ValueError):
        raise ObserverError("could not read the secret key") from None
    if len(key) < 32:
        raise ObserverError("the secret key is invalid")
    return key


def pseudonym(key: bytes, domain: bytes, raw: str, prefix: str) -> str:
    digest = hmac.new(key, domain + raw.encode("utf-8", "surrogatepass"), hashlib.sha256)
    return f"{prefix}_{digest.hexdigest()[:32]}"


# --- collect ---------------------------------------------------------------------------------

def normalize_tool(name: object) -> str:
    if not isinstance(name, str):
        return "other"
    if name in BUILTIN_TOOLS:
        return name
    if name.startswith("mcp__"):
        return "mcp"
    return "other"


def build_record(payload: object, state: dict, key: bytes) -> dict | None:
    """Map a hook payload to an allowlisted record without `sequence`; None if ignored."""
    if not isinstance(payload, dict):
        raise ObserverError("input is not a JSON object")
    event_type = HOOK_EVENTS.get(payload.get("hook_event_name"))
    if event_type is None:
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    tool, tool_call_id = "none", None
    if event_type in TOOL_EVENTS:
        tool = normalize_tool(payload.get("tool_name"))
        raw_call = payload.get("tool_use_id")
        if isinstance(raw_call, str) and raw_call:
            tool_call_id = pseudonym(key, _CALL_DOMAIN, raw_call, "call")
    now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return {
        "schema": SCHEMA,
        "event_id": str(uuid.uuid4()),
        "operator_id": state["operator_id"],
        "agent_id": state["agent_id"],
        "configuration_id": state["configuration_id"],
        "run_id": pseudonym(key, _RUN_DOMAIN, session_id, "run"),
        "sequence": 0,
        "occurred_at": now,
        "source": SOURCE,
        "event_type": event_type,
        "tool": tool,
        "evidence_level": EVIDENCE_LEVEL,
        "configuration_basis": CONFIGURATION_BASIS,
        "tool_call_id": tool_call_id,
    }


def _load_sequences(state_dir: Path) -> dict[str, int]:
    p = _paths(state_dir)
    try:
        data = json.loads(p["sequences"].read_text())
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, int) and v > 0}
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        pass
    # Counter missing or damaged: rebuild from the log so sequences never repeat.
    seqs: dict[str, int] = {}
    if p["events"].exists():
        with p["events"].open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    run, seq = rec["run_id"], rec["sequence"]
                except (ValueError, KeyError, TypeError):
                    continue
                if isinstance(run, str) and isinstance(seq, int) and seq > seqs.get(run, 0):
                    seqs[run] = seq
    return seqs


def collect(state_dir: Path, raw: bytes) -> dict | None:
    """Append one record for a hook payload. Returns the record, or None if ignored."""
    if len(raw) > MAX_INPUT_BYTES:
        raise ObserverError("input exceeds the 1 MB limit; ignored")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ObserverError("input is not valid JSON; ignored") from None
    state = _read_state(state_dir)
    key = _read_secret(state_dir)
    record = build_record(payload, state, key)
    if record is None:
        return None
    p = _paths(state_dir)
    with _locked(state_dir):
        seqs = _load_sequences(state_dir)
        record["sequence"] = seqs.get(record["run_id"], 0) + 1
        seqs[record["run_id"]] = record["sequence"]
        # Counter first: a crash leaves a gap in sequence, never a duplicate.
        _write_private(p["sequences"], json.dumps(seqs))
        fd = os.open(p["events"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)
    return record


# --- export ----------------------------------------------------------------------------------

def validate_record(rec: object, state: dict) -> bool:
    """True only for a record with exactly the allowlisted fields and valid values."""
    if not isinstance(rec, dict) or set(rec) != set(FIELDS):
        return False
    s = rec["sequence"]
    call = rec["tool_call_id"]

    def is_str(v: object, pattern: re.Pattern) -> bool:
        return isinstance(v, str) and pattern.fullmatch(v) is not None

    try:
        uuid_ok = str(uuid.UUID(rec["event_id"])) == rec["event_id"]
    except (ValueError, TypeError, AttributeError):
        uuid_ok = False
    try:
        time_ok = is_str(rec["occurred_at"], TIME_RE) and bool(
            datetime.fromisoformat(rec["occurred_at"])
        )
    except ValueError:
        time_ok = False
    tool_event = isinstance(rec["event_type"], str) and rec["event_type"] in TOOL_EVENTS
    return (
        rec["schema"] == SCHEMA
        and uuid_ok
        and time_ok
        and rec["operator_id"] == state["operator_id"]
        and rec["agent_id"] == state["agent_id"]
        and is_str(rec["configuration_id"], CONFIG_RE)
        and is_str(rec["run_id"], RUN_RE)
        and isinstance(s, int) and not isinstance(s, bool) and s > 0
        and rec["source"] == SOURCE
        and isinstance(rec["event_type"], str) and rec["event_type"] in EVENT_TYPES
        and isinstance(rec["tool"], str) and rec["tool"] in TOOLS
        and (rec["tool"] != "none") == tool_event
        and rec["evidence_level"] == EVIDENCE_LEVEL
        and rec["configuration_basis"] == CONFIGURATION_BASIS
        and (call is None or (tool_event and is_str(call, CALL_RE)))
    )


def export(state_dir: Path, out) -> tuple[int, int]:
    """Write revalidated records to `out`. Returns (written, dropped)."""
    state = _read_state(state_dir)
    events = _paths(state_dir)["events"]
    written = dropped = 0
    seen: set[str] = set()
    if not events.exists():
        return 0, 0
    with _locked(state_dir), events.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                rec = None
            if not validate_record(rec, state) or rec["event_id"] in seen:
                dropped += 1
                continue
            seen.add(rec["event_id"])
            clean = {k: rec[k] for k in FIELDS}  # rebuilt in fixed order from the allowlist
            out.write(json.dumps(clean, separators=(",", ":")) + "\n")
            written += 1
    return written, dropped


# --- CLI -------------------------------------------------------------------------------------

def _warn(msg: str) -> None:
    print(f"ajantik observer: warning: {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ajantik.observer")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="Create the state, or update the configuration label.")
    p_init.add_argument("--state-dir", type=Path, required=True)
    p_init.add_argument("--configuration-id", required=True)
    for name, text in (("collect", "record the hook payload on stdin"), ("export", "sanitized JSONL")):
        sub.add_parser(name, help=text).add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    state_dir = args.state_dir.expanduser()

    if args.command == "collect":
        # Fail open: always exit 0 with empty stdout so the hook never blocks or decides.
        try:
            collect(state_dir, sys.stdin.buffer.read(MAX_INPUT_BYTES + 1))
        except ObserverError as e:
            _warn(str(e))
        except Exception as e:  # noqa: BLE001 - never let the hook fail; never print payload
            _warn(f"unexpected error ({type(e).__name__}); event not recorded")
        return 0

    try:
        if args.command == "init":
            state = init_state(state_dir, args.configuration_id)
            print(f"operator_id: {state['operator_id']}")
            print(f"agent_id: {state['agent_id']}")
            print(f"configuration_id: {state['configuration_id']} (declared; not measured)")
            print(f"events: {_paths(state_dir)['events']}")
            return 0
        written, dropped = export(state_dir, sys.stdout)
        if dropped:
            _warn(f"{dropped} invalid or duplicate records left out")
        print(f"ajantik observer: {written} records exported", file=sys.stderr)
        return 0
    except ObserverError as e:
        _warn(str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
