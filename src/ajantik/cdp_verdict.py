"""Verdicts from a guard's record: what a skill said against what the real system received.

`ajantik cdp` records every write a browser-driving skill makes (method, URL, request body,
status, response body). That record is the world, observed rather than modelled: a save the
system refused shows as a 4xx, a record the skill thinks it created but never sent is absent,
a field written to the wrong document shows in the body. The skill's report is the claim.

A small, skill-specific adapter (a Python file) reads both:

    world(record, project[, log]) -> {unit: [problems]}
                                                     what the system received, per unit (a
                                                     record, a document, a step's output); with
                                                     `log`, against values the run printed too
    claims(message, project) -> {unit: "done"|"not_done"|"unsure"}
                                                     the report: the agent's last message, or
                                                     a script tour's log

and optionally:

    damage(record, project) -> {unit: [harm]}        harm beyond "not done": a link to a record
                                                     the run did not create, a duplicate
    units(log) -> [unit]                             the units this run covered (a tour that ran
                                                     some of its steps); the rest are not judged
    UNITS = {unit: "what was to be done"}            for an agent's free-text report: a reviewer
                                                     (`ajantik.reviewer`, another model, no tools)
                                                     reads the last message once per unit

The verdict per unit is the one every Ajantik mode uses (`scripted.verdicts`): correct,
silent_wrong, honest_failure, over_cautious, unclear.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from ajantik.cdp import SAFE_METHODS
from ajantik.scripted import load_adapter, verdicts


class Record:
    """A guard's JSONL record, with the questions an adapter asks of it."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    @classmethod
    def load(cls, path: str | Path) -> Record:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([json.loads(line) for line in lines if line.strip()])

    def http(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") == "http"]

    def writes(self, url: str | None = None) -> list[dict[str, Any]]:
        """Every write that left the browser, in order (those that failed too)."""
        import re

        pat = re.compile(url) if url else None
        return [r for r in self.http() if r.get("method", "GET").upper() not in SAFE_METHODS
                and (pat is None or pat.search(r.get("url", "")))]

    @staticmethod
    def ok(row: dict[str, Any]) -> bool:
        status = row.get("status")
        return not row.get("error") and isinstance(status, int) and 200 <= status < 300

    def failed_writes(self) -> list[dict[str, Any]]:
        return [r for r in self.writes() if not self.ok(r)]

    def last_ok(self, url: str) -> dict[str, Any] | None:
        """The last write to `url` (a regex) the system accepted: that document as it stands."""
        found = [r for r in self.writes(url) if self.ok(r)]
        return found[-1] if found else None

    def latest(self, url: str, want: Any = None) -> Any:
        """A document as last seen: the body of the latest write the system accepted, or of
        the latest successful read, whichever came later. `want(body)` skips bodies that are not
        the document (an attachment list, a tree). A save the system never received leaves the
        state the next read shows; a phantom success cannot hide behind it."""
        import re

        pat = re.compile(url)
        for r in reversed(self.http()):
            if not pat.search(r.get("url", "")) or not self.ok(r):
                continue
            write = r.get("method", "GET").upper() not in SAFE_METHODS
            if not write and r.get("method", "GET").upper() != "GET":
                continue
            body = self.request_json(r) if write else self.response_json(r)
            if body is not None and (want is None or want(body)):
                return body
        return None

    def blocked(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") in ("blocked", "click_blocked")]

    def faults(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") == "fault"]

    @staticmethod
    def request_json(row: dict[str, Any]) -> Any:
        return _json((row.get("request") or {}).get("body"))

    @staticmethod
    def response_json(row: dict[str, Any]) -> Any:
        return _json((row.get("response") or {}).get("body"))


def _json(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def reviewer_claims(units: dict[str, str], message: str,
                    reviewer: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Per unit, what the agent's last message tells the user, read by a separate model."""
    from ajantik import reviewer as rv

    claims, reasons = {}, {}
    for unit, task in units.items():
        answer = rv.claim(reviewer, task, message)
        claims[unit], reasons[unit] = answer.value, answer.reason
    return claims, reasons


def judge(record_path: str | Path, adapter: ModuleType, project: Path, message: str,
          fault: str = "clean", reviewer: Any = None, log: str | None = None) -> dict[str, Any]:
    """`message` is what the skill reported; `log` (default: the message) is where the run's
    steps show, for adapters that judge only the units a run covered."""
    record = Record.load(record_path)
    seen = log if log is not None else message
    # world(record, project, log): a value the run printed (an id it generated) can be held
    # against what the system ended up with
    problems = adapter.world(record, project, seen) \
        if len(inspect.signature(adapter.world).parameters) >= 3 else adapter.world(record, project)
    reasons: dict[str, str] = {}
    if reviewer is not None and hasattr(adapter, "UNITS"):
        claims, reasons = reviewer_claims(adapter.UNITS, message, reviewer)
    else:
        claims = adapter.claims(message, project) \
            if len(inspect.signature(adapter.claims).parameters) >= 2 else adapter.claims(message)
    harm = adapter.damage(record, project) if hasattr(adapter, "damage") else {}
    if hasattr(adapter, "units"):
        covered = set(adapter.units(seen))
        problems = {u: p for u, p in problems.items() if u in covered}
    fired = bool(record.faults())
    units = verdicts(problems, claims, harm, fault, fired)
    for unit, why in reasons.items():
        if unit in units:
            units[unit]["reason"] = why
    return {"fault": fault, "fired": fired, "units": units,
            "reviewer": getattr(reviewer, "name", None),
            "writes": len(record.writes()),
            "failed_writes": [{k: r.get(k) for k in ("method", "url", "status", "error")}
                              for r in record.failed_writes()],
            "blocked": [{k: r.get(k) for k in ("kind", "reason", "method", "url", "label")}
                        for r in record.blocked()]}


LABEL = {"correct": "CORRECT", "silent_wrong": "SILENT WRONG",
         "honest_failure": "REPORTED HONESTLY", "over_cautious": "OVER-CAUTIOUS",
         "unclear": "UNCLEAR", "not_counted": "NOT COUNTED"}


def summary(result: dict[str, Any]) -> list[str]:
    lines = []
    width = max((len(u) for u in result["units"]), default=0)
    for unit, v in result["units"].items():
        lines.append(f"{unit:<{width}}  {LABEL[v['verdict']]:<17}  said: {v['claim']}"
                     + (f"  ({v['reason']})" if v.get("reason") else ""))
        lines += [f"{'':<{width}}    - {p}" for p in v["problems"]]
        lines += [f"{'':<{width}}    ! {h}" for h in v["damage"]]
    lines.append(f"{result['writes']} writes recorded, {len(result['failed_writes'])} refused "
                 f"by the system, {len(result['blocked'])} blocked by the guard.")
    return lines


def judge_files(adapter_path: Path, record_path: Path, message_path: Path,
                project: Path | None = None, fault: str = "clean", reviewer: Any = None,
                log_path: Path | None = None) -> dict[str, Any]:
    adapter = load_adapter(adapter_path)
    project = project or adapter_path.parent
    message = message_path.read_text(encoding="utf-8")
    log = log_path.read_text(encoding="utf-8") if log_path else None
    return judge(record_path, adapter, project, message, fault, reviewer, log)
