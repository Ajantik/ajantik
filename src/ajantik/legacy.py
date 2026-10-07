"""Compatibility with older file formats.

The lab that develops this tool keeps records in formats it used before; it maps their
spellings here. The public release has no such files, so every map is empty.
"""

from __future__ import annotations

FAULT_FAMILIES: dict[str, str] = {}
ANOMALY_PREFIX = ""
ANOMALY_NAMES: dict[str, str] = {}
FAULT_MODULES: dict[str, str] = {}
LABELS: dict[str, str] = {}
FILE_KEYS: dict[str, str] = {}
SCENARIO_KEYS: dict[str, str] = {}
EFFECTS: dict[str, str] = {}
OLD_TOOL_KEY = "\0"  # matches no key
OLD_TOOL_FIELDS = ("field", "value")
TRIAL_FIELDS: dict[str, str] = {}
MANIFESTS: tuple[str, ...] = ()
RECORD_DIRS: tuple[str, ...] = ()
GUARD_SPEC_KEYS: dict[str, str] = {}
GUARD_SEED_KEYS: dict[str, str] = {}
GUARD_KINDS: dict[str, str] = {}
GUARD_SYNONYM_SETS: dict[str, str] = {}
GUARD_FAMILIES: dict[str, str] = {}
