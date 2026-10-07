"""Run a guard spec: generate variants, call the guard, report where its decision is wrong.

Spec (YAML):
  guard: js:file.js#function             # or py:file.py#function
  kind: decision                         # decision: f(text) -> value | comparison: f(a, b) -> same?
  seeds:                                 # decision
    - {text: "Submit the form", expected: true}
  codes: ["H360F", "H350"]               # comparison
  synonyms: default                      # optional: default | en | tr | {word: [synonyms]}
  exclude: [ascii_letters]               # optional: families that change the meaning for this guard
  family_expected: {lookalike_letter: true}  # optional: fixed expected value for a whole family

Older lab specs used other key, kind and family names; ajantik.legacy maps them.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ajantik import legacy
from ajantik.guards.callers import Guard
from ajantik.guards.mutations import (
    EMBED,
    FAMILY_ALIASES,
    INVARIANCE,
    SYNONYM_SETS,
    near_misses,
    synonym_variants,
)

# Alias data: old spec vocabulary -> current names.
SPEC_KEY_ALIASES = legacy.GUARD_SPEC_KEYS
SEED_KEY_ALIASES = legacy.GUARD_SEED_KEYS
KIND_ALIASES = legacy.GUARD_KINDS
SYNONYM_SET_ALIASES = legacy.GUARD_SYNONYM_SETS

FIX_HINTS = {
    "decomposed_unicode": "Apply unicodedata/String.normalize('NFKC') before comparing.",
    "fullwidth": "NFKC normalisation turns full-width letters into plain letters.",
    "zero_width": "Remove format characters (Unicode category Cf).",
    "soft_hyphen": "Remove format characters (Unicode category Cf).",
    "no_break_space": "Collapse every kind of space (\\s and NBSP) into a single space.",
    "upper_case": "Lower-case with Turkish-aware rules (dotted capital I to i, I to dotless i), "
                  "or fold to ASCII before comparing.",
    "lower_case": "Lower-case with Turkish-aware rules (dotted capital I to i, I to dotless i), "
                  "or fold to ASCII before comparing.",
    "ascii_letters": "Fold Turkish letters to their ASCII equivalents before comparing.",
    "lookalike_letter": "Treat text that mixes alphabets (Latin + Cyrillic) as suspect, or reject it.",
    "extra_whitespace": "Trim leading/trailing spaces and collapse runs of spaces into one.",
    "punctuation": "Match on words or a substring, not on the whole text.",
    "embedded_text": "Search for a substring in joined texts too.",
    "near_code": "Compare the code exactly, including its suffixes and their case.",
    "seed": "The guard does not give the expected decision even on the base example.",
}


@dataclass
class Case:
    family: str
    seed: str
    variant: str
    expected: Any
    got: Any = None
    error: str = ""
    note: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and self.got == self.expected


def visible(s: str) -> str:
    """Make invisible and look-alike characters visible in reports."""
    out = []
    for c in s:
        cat = unicodedata.category(c)
        if c == "\n":
            out.append("⏎")
        elif cat in ("Cf", "Mn") or c == "\u00a0" or (
            ord(c) > 0x7F and cat.startswith("L")
            and unicodedata.name(c, "").startswith(("CYRILLIC", "FULLWIDTH"))
        ):
            out.append(f"⟨U+{ord(c):04X}⟩")
        else:
            out.append(c)
    return "".join(out)


def _renamed(data: dict, aliases: dict[str, str], where: str) -> dict:
    """Copy `data` with old keys renamed. Giving both the old and the new key is an error."""
    out: dict = {}
    for key, value in data.items():
        new = aliases.get(key, key)
        if new in out:
            raise ValueError(f"{where}: '{key}' and '{new}' mean the same thing; give only one.")
        out[new] = value
    return out


def _family(name: str) -> str:
    return FAMILY_ALIASES.get(name, name)


def normalise_spec(raw: dict) -> dict:
    """Return the spec with current key, kind, family and synonym-set names."""
    if not isinstance(raw, dict):
        raise TypeError("A guard spec must be a YAML mapping.")
    spec = _renamed(raw, SPEC_KEY_ALIASES, "spec")
    if "guard" not in spec:
        raise ValueError("The spec has no 'guard' (e.g. 'py:file.py#function').")
    kind = spec.get("kind", "decision")
    spec["kind"] = KIND_ALIASES.get(kind, kind)
    if "seeds" in spec:
        spec["seeds"] = [_renamed(s, SEED_KEY_ALIASES, "seed") for s in spec["seeds"] or []]
    spec["exclude"] = [_family(f) for f in spec.get("exclude") or []]
    spec["family_expected"] = {_family(f): v for f, v in (spec.get("family_expected") or {}).items()}
    synonyms = spec.get("synonyms")
    if isinstance(synonyms, str):
        spec["synonyms"] = SYNONYM_SET_ALIASES.get(synonyms, synonyms)
    return spec


def _families(spec: dict) -> tuple[set[str], dict[str, Any]]:
    return set(spec["exclude"]), dict(spec["family_expected"])


def _synonyms(spec: dict) -> dict[str, list[str]]:
    synonyms = spec.get("synonyms")
    if isinstance(synonyms, str):
        if synonyms not in SYNONYM_SETS:
            raise ValueError(f"Unknown synonym set: {synonyms} (known: {', '.join(SYNONYM_SETS)})")
        return SYNONYM_SETS[synonyms]
    return synonyms or {}


def _decision_cases(spec: dict) -> list[Case]:
    cases: list[Case] = []
    skip, fixed = _families(spec)
    synonyms = _synonyms(spec)
    for seed in spec["seeds"]:
        text, expected = str(seed["text"]), seed["expected"]
        cases.append(Case("seed", text, text, expected))
        for m in INVARIANCE + ([EMBED] if expected else []):
            v = m.fn(text)
            if m.name not in skip and v is not None and v != text:
                cases.append(Case(m.name, text, v, fixed.get(m.name, expected)))
        for alt, v in synonym_variants(text, synonyms):
            cases.append(Case("synonym", text, v, expected, note=alt))
    return cases


def _comparison_cases(spec: dict) -> list[Case]:
    cases: list[Case] = []
    skip, fixed = _families(spec)
    for code in map(str, spec["codes"]):
        cases.append(Case("seed", code, code, True))
        for m in INVARIANCE:
            v = m.fn(code)
            if m.name not in skip and v is not None and v != code:
                cases.append(Case(m.name, code, v, fixed.get(m.name, True)))
        for why, v in near_misses(code):
            cases.append(Case("near_code", code, v, False, note=why))
    return cases


def run_spec(spec_path: Path) -> tuple[list[Case], dict]:
    spec = normalise_spec(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    guard = Guard(spec["guard"], spec_path.parent)
    kind = spec["kind"]
    if kind == "decision":
        cases = _decision_cases(spec)
        results = guard.call_many([[c.variant] for c in cases])
    elif kind == "comparison":
        cases = _comparison_cases(spec)
        results = guard.call_many([[c.seed, c.variant] for c in cases])
    else:
        raise ValueError(f"Unknown guard kind: {kind} (expected 'decision' or 'comparison')")
    for c, r in zip(cases, results):
        if r["ok"]:
            c.got = bool(r["value"]) if kind == "comparison" else r["value"]
        else:
            c.error = r["error"]
    return cases, spec


def report(spec_path: Path, spec: dict, cases: list[Case]) -> str:
    spec = normalise_spec(spec)
    by_family: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        by_family[c.family].append(c)
    policy = by_family.pop("synonym", [])
    failed = [c for fam in by_family.values() for c in fam if not c.passed]
    total = sum(len(v) for v in by_family.values())
    lines = [
        f"# Guard fuzzing report: `{spec['guard']}`",
        "",
        (
            f"Kind: {spec['kind']} · {total} variants tried · "
            f"**wrong decision on {len(failed)}** · synonym policy checks: {len(policy)}"
        ),
        "",
        "| Family | Tried | Wrong decision | What was tried |",
        "|---|---|---|---|",
    ]
    descriptions = {m.name: m.description for m in INVARIANCE} | {
        EMBED.name: "Joined with other text", "seed": "Base example (no variant)",
        "near_code": "A code that looks similar but means something else",
    }
    for fam, cs in by_family.items():
        bad = sum(not c.passed for c in cs)
        mark = "⚠\ufe0f " if bad else ""
        lines.append(f"| {fam} | {len(cs)} | {mark}{bad} | {descriptions.get(fam, '')} |")
    if failed:
        lines += ["", "## Wrong decisions (at most 3 examples per family)", ""]
        shown: dict[str, int] = defaultdict(int)
        for c in failed:
            if shown[c.family] >= 3:
                continue
            shown[c.family] += 1
            got = f"error: {c.error}" if c.error else repr(c.got)
            note = f" ({c.note})" if c.note else ""
            lines.append(
                f"- **{c.family}**{note}: `{visible(c.seed)}` → `{visible(c.variant)}` · "
                f"expected {c.expected!r}, guard {got}"
            )
        lines += ["", "## Fix hints", ""]
        for fam in dict.fromkeys(c.family for c in failed):
            lines.append(f"- **{fam}:** {FIX_HINTS.get(fam, '')}")
    if policy:
        diff = [c for c in policy if not c.passed]
        lines += [
            "",
            "## Synonyms (a policy question, not counted as errors)",
            "",
            (
                f"For {len(diff)} of {len(policy)} synonyms the guard decided differently from the "
                "base example. Whether they should be treated the same is for the guard's owner "
                "to decide."
            ),
            "",
        ]
        lines += [f"- `{c.note}`: `{c.variant}` → guard {c.got!r}, base example {c.expected!r}"
                  for c in diff[:12]]
    return "\n".join(lines) + "\n"


def save(out_dir: Path, name: str, text: str, cases: list[Case]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{name}.md"
    md.write_text(text, encoding="utf-8")
    (out_dir / f"{name}.json").write_text(
        json.dumps([asdict(c) | {"passed": c.passed} for c in cases], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return md
