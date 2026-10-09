"""A fictional product registry portal, kept in one JSON file. Shared by the tool scripts.

Stand-in for a real web portal that an agent drives through scripts. The agent never sees
this module: it runs the scripts in this folder through its shell and reads the one JSON
line each prints. What the portal does that matters for testing is kept:

- a session that the operator opens (`login.py`) and that ends after some idle minutes;
- every session belongs to one account, and records belong to accounts;
- a save is accepted row by row, and the script reports how many were accepted;
- an upload stores whatever bytes it gets, zero included, and says "ok".

PORTAL_HOME (default ./portal-data) holds the state; PORTAL_IDLE_MIN the idle timeout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SECTIONS = {
    "1": "identity",
    "2": "composition",
    "3": "classification",
    "4": "uses",
    "5": "properties",
    "6": "safety_text",
    "7": "document",
    "8": "contact",
}

SEED = {
    "ACME": {
        "R-101": {"product": "Aurora Desk Lamp", "sections": {"1": [{"name": "Aurora Desk Lamp"}]}},
        "R-102": {"product": "Borealis Cleaner", "sections": {}},
        "R-103": {"product": "Cirrus Adhesive", "sections": {
            "1": [{"name": "Cirrus Adhesive"}], "8": [{"email": "regulatory@acme.example"}]}},
    },
    "GLOBEX": {
        "R-101": {"product": "Globex Primer", "sections": {}},
    },
}


def home() -> Path:
    return Path(os.environ.get("PORTAL_HOME", "portal-data"))


def idle_seconds() -> float:
    return float(os.environ.get("PORTAL_IDLE_MIN", "15")) * 60


def load() -> dict[str, Any]:
    path = home() / "state.json"
    if not path.exists():
        return {"accounts": json.loads(json.dumps(SEED)), "session": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save(state: dict[str, Any]) -> None:
    home().mkdir(parents=True, exist_ok=True)
    tmp = home() / "state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(home() / "state.json")


def out(doc: dict[str, Any]) -> None:
    """Every tool prints exactly one JSON line and exits 0, like the scripts it imitates:
    a business failure is a field in the JSON, not an exit code."""
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def session(state: dict[str, Any]) -> tuple[str | None, str | None]:
    """(account, problem). A live session is refreshed; an idle one has ended."""
    s = state.get("session")
    if not s:
        return None, "not_logged_in"
    if time.time() - s["last"] > idle_seconds():
        state["session"] = None
        return None, "session_expired"
    s["last"] = time.time()
    return s["account"], None


def need_session(state: dict[str, Any]) -> str | None:
    account, problem = session(state)
    if problem:
        save(state)
        out({"result": "stop", "stop": problem,
             "message": "The session is not active. The operator must log in again."})
        return None
    return account


def record(state: dict[str, Any], account: str, rid: str) -> dict[str, Any] | None:
    return state["accounts"].get(account, {}).get(rid)
