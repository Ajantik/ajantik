"""Load a Claude skill (SKILL.md standard) from a directory or zip."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from ajantik.identity import recipe_files, recipe_fingerprint


@dataclass
class Skill:
    name: str
    description: str
    body: str
    files: dict[str, bytes]
    fingerprint: str
    co: list[Skill] = field(default_factory=list)  # other skills installed next to this one

    def extra_files(self) -> list[str]:
        return sorted(f for f in self.files if f != "SKILL.md")

    def read(self, rel: str) -> str:
        if rel not in self.files:
            raise KeyError(rel)
        return self.files[rel].decode("utf-8", errors="replace")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    _, front, body = text.split("---", 2)
    return yaml.safe_load(front) or {}, body.lstrip("\n")


def load_skill(path: Path) -> Skill:
    files = recipe_files(path)
    if "SKILL.md" not in files:
        raise ValueError(f"SKILL.md not found: {path}")
    meta, body = _split_frontmatter(files["SKILL.md"].decode("utf-8"))
    if not meta.get("name") or not meta.get("description"):
        raise ValueError("The SKILL.md header must have 'name' and 'description'")
    return Skill(
        name=str(meta["name"]),
        description=str(meta["description"]),
        body=body,
        files=files,
        fingerprint=recipe_fingerprint(files),
    )


def combine(main: Skill, extras: list[Skill]) -> Skill:
    """Install extra skills next to the main one (drug-interaction trials).

    Extra skills' files are namespaced by skill name, so the combination gets its own content
    fingerprint and therefore its own identity and track record.
    """
    if not extras:
        return main
    files = dict(main.files)
    for x in extras:
        if x.name == main.name or any(f.startswith(f"{x.name}/") for f in main.files):
            raise ValueError(f"Skill name clash: {x.name}")
        files.update({f"{x.name}/{rel}": data for rel, data in x.files.items()})
    return replace(main, files=files, fingerprint=recipe_fingerprint(files), co=list(extras))
