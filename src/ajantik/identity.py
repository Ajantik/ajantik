"""Agent identity: a content fingerprint of the recipe plus the setup that runs it.

A track record belongs to an identity, never to a name. Change one word in the recipe, the model
or the harness and it is a different identity with its own track record.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

IGNORED_PARTS = {"__MACOSX", ".DS_Store", "__pycache__", ".git"}


def _normalize(data: bytes) -> bytes:
    """Line endings must not change the fingerprint of a text file."""
    if b"\0" in data:
        return data
    return data.replace(b"\r\n", b"\n")


def _strip_common_root(names: list[str]) -> list[str]:
    """A zip of `skill/` and a zip of the files inside it are the same recipe."""
    parts = [n.split("/") for n in names]
    while parts and all(len(p) > 1 for p in parts) and len({p[0] for p in parts}) == 1:
        parts = [p[1:] for p in parts]
    return ["/".join(p) for p in parts]


def recipe_files(path: Path) -> dict[str, bytes]:
    """Relative path -> normalized content for a skill directory or zip."""
    raw: dict[str, bytes] = {}
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            rel = f.relative_to(path)
            if f.is_file() and not IGNORED_PARTS.intersection(rel.parts):
                raw[rel.as_posix()] = f.read_bytes()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or IGNORED_PARTS.intersection(info.filename.split("/")):
                    continue
                raw[info.filename] = z.read(info)
    else:
        raise ValueError(f"Expected a skill directory or zip file: {path}")
    if not raw:
        raise ValueError(f"The recipe has no files: {path}")
    names = sorted(raw)
    return {new: _normalize(raw[old]) for old, new in zip(names, _strip_common_root(names))}


def recipe_fingerprint(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        content = files[name]
        h.update(name.encode() + b"\0" + str(len(content)).encode() + b"\0" + content)
    return h.hexdigest()


@dataclass(frozen=True)
class Identity:
    recipe: str
    model: str
    effort: str
    harness: str

    @property
    def id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict[str, str]:
        return {**asdict(self), "id": self.id}
