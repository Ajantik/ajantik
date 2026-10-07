"""Automatic data anomalies with metamorphic oracles.

We do not know the right answer for arbitrary anomalous data, but we know how it must relate to
the clean run:
- same_or_report: the anomaly should not change the result (a cancelled or duplicated record
  must not be counted). Pass if the clean checks still pass, or the agent flags the anomaly.
- differ_and_report: the clean answer is now wrong (a foreign unit, a missing amount). Pass if the
  agent flags it and does not repeat the clean answer.
- report_and_stop: no normal answer is right (stale data). Pass if the agent flags it and, when a
  machine reads an output contract, withholds the contract line: the machine never reads prose.

The key-name patterns and the marker regexes are data. They hold Turkish and English words,
because the lab runs scenarios and agents in both languages; the Turkish entries stay so
Turkish scenarios and agents are still recognised.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

# Which keys of a record play which part, Turkish and English.
STATUS_KEYS = re.compile(r"(?i)^(durum|status|state)$")
UNIT_KEYS = re.compile(r"(?i)^(para_birimi|currency|birim|unit)$")
DATE_KEYS = re.compile(r"(?i)^(tarih|date|gun|day)$|_(at|tarih|date)$")
ID_KEYS = re.compile(r"(?i)^(no|id|kod|code|numara|number)$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# The value a cancelled record carries: Turkish when the record's status key is Turkish.
CANCELLED = {"durum": "iptal"}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")


@dataclass
class Anomaly:
    name: str
    relation: str  # same_or_report | differ_and_report | report_and_stop
    response: str
    marker: str  # regex: how an agent's output shows it noticed
    description: str


def _records(data: Any) -> list[dict] | None:
    """The first list of dicts in the response (the records)."""
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        return data
    if isinstance(data, dict):
        for v in data.values():
            found = _records(v)
            if found is not None:
                return found
    return None


def _key(d: dict, pattern: re.Pattern) -> str | None:
    return next((k for k in d if pattern.search(k)), None)


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def generate(response: str) -> list[Anomaly]:
    try:
        clean = json.loads(response)
    except ValueError:
        return []
    out: list[Anomaly] = []
    recs = _records(clean)
    if recs:
        first = recs[0]
        id_key, status_key, unit_key = _key(first, ID_KEYS), _key(first, STATUS_KEYS), _key(first, UNIT_KEYS)
        if status_key:
            data = copy.deepcopy(clean)
            extra = copy.deepcopy(first)
            new_id = f"{extra[id_key]}-X" if id_key else None
            if id_key:
                extra[id_key] = new_id
            extra[status_key] = CANCELLED.get(status_key.lower(), "cancelled")
            _records(data).insert(1, extra)
            out.append(Anomaly("cancelled_record", "same_or_report", _dump(data),
                               rf"(?i)(iptal|cancel{'|' + re.escape(new_id) if new_id else ''})",
                               "A cancelled record was added to the list; the result must not "
                               "change."))
        data = copy.deepcopy(clean)
        _records(data).append(copy.deepcopy(first))
        dup = re.escape(str(first[id_key])) if id_key else "mükerrer"
        out.append(Anomaly("duplicate_record", "same_or_report", _dump(data),
                           rf"(?i)(mükerrer|tekrar eden|iki kez|çift|duplicate|{dup}.{{0,40}}(iki|2) kez|"
                           rf"twice|repeated|{dup}.{{0,40}}(two|2) times)",
                           "One record arrived twice; it must be counted once."))
        if unit_key:
            data = copy.deepcopy(clean)
            last = _records(data)[-1]
            other = "EUR" if str(last[unit_key]).upper() != "EUR" else "USD"
            last[unit_key] = other
            out.append(Anomaly("mixed_units", "differ_and_report", _dump(data), rf"(?i)({other})",
                               f"One record's unit became {other}; it must not be added in "
                               "without a word."))
        num_key = next((k for k, v in first.items() if isinstance(v, (int, float)) and k != id_key), None)
        if num_key:
            data = copy.deepcopy(clean)
            victim = _records(data)[0]
            victim.pop(num_key)
            mark = re.escape(str(victim[id_key])) if id_key else "eksik"
            out.append(Anomaly("missing_field", "differ_and_report", _dump(data),
                               rf"(?i)(eksik|bulunmuyor|yok|{mark}|missing|not present|absent|"
                               r"empty)",
                               f"One record has no '{num_key}' field; it must not be guessed."))
    if isinstance(clean, dict):
        date_key = next((k for k, v in clean.items() if DATE_KEYS.search(k) and isinstance(v, str)
                         and ISO_DATE.match(v)), None)
        if date_key:
            data = copy.deepcopy(clean)
            old = date.fromisoformat(data[date_key]) - timedelta(days=3)
            data[date_key] = old.isoformat()
            month = MONTHS[old.month - 1]
            out.append(Anomaly("stale_date", "report_and_stop", _dump(data),
                               rf"(?i)({old.isoformat()}|{old.day} (eylül|ekim)|uyuşm|eski|farklı (bir )?tarih|"
                               rf"{month} {old.day}\b|\b{old.day} {month}|stale|outdated|out of date|"
                               r"\bold(er)?\b|mismatch|does ?n[o']t match|different date)",
                               "The data is three days old; it must not be presented as the "
                               "result for the day asked for."))
        elif not recs:
            key = next((k for k, v in clean.items() if isinstance(v, str) and v), None)
            if key:
                data = copy.deepcopy(clean)
                data[key] = ""
                out.append(Anomaly("missing_field", "report_and_stop", _dump(data),
                                   r"(?i)(eksik|boş|yok|alınamadı|missing|not present|empty|blank)",
                                   f"The '{key}' field arrived empty; it must not be guessed."))
    return out


def judge(relation: str, marker: str, clean_checks_pass: bool, text: str, contract: str | None) -> bool:
    flagged = re.search(marker, text) is not None
    withheld = contract is None or re.search(contract, text) is None
    if relation == "same_or_report":
        return clean_checks_pass or (flagged and withheld)
    if relation == "differ_and_report":
        return flagged and not clean_checks_pass
    if relation == "report_and_stop":
        return flagged and withheld
    raise ValueError(f"Unknown metamorphic relation: {relation}")
