"""What the lab knows about each agent it has tested: how to find its closing message.

The judge reads one thing, the agent's closing message, and each agent prints it
differently. Goose prints a banner, separators and tool echoes around it; Claude Code in
`--print --output-format json` mode prints one JSON object whose `result` is the message.
Reading one agent's output with another's rules feeds the judge the wrong text, and the
judge's measured agreement does not carry over. So extraction is per agent, and an agent
with no profile is extracted generically and flagged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ajantik.labelling import closing_message as _goose_closing


def _never_failed(stdout: str) -> str | None:
    return None


@dataclass(frozen=True)
class Profile:
    name: str
    extract: Callable[[str, list[str]], str]
    validated: bool  # extraction checked against real transcripts of this agent
    # The agent's own infrastructure failed (authentication, API error): the run says
    # nothing about its behaviour, and its "I could not authenticate" must never be scored
    # as an honest report. Returns the reason, or None.
    infra_failure: Callable[[str], str | None] = _never_failed


def _claude_code(stdout: str, replies: list[str]) -> str:
    """`claude -p --output-format json` prints one JSON object; `result` is the message.
    Plain `--output-format text` prints only the message, so it is taken as is."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("result"), str):
            return doc["result"].strip()
    return stdout.strip()


def _claude_code_failure(stdout: str) -> str | None:
    for line in reversed(stdout.strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("type") == "result":
            if doc.get("terminal_reason") == "api_error" or (
                    doc.get("is_error") and not doc.get("num_turns", 0) > 1
                    and not (doc.get("usage") or {}).get("output_tokens")):
                return f"agent infrastructure: {doc.get('result') or doc.get('terminal_reason')}"
            return None
    return None


def _generic(stdout: str, replies: list[str]) -> str:
    return _goose_closing(stdout, replies)


PROFILES = {
    "goose": Profile("goose", _goose_closing, validated=True),
    # Validated on 40 real trials (lab experiment 012): `result` held the closing message
    # every time, and a blind human agreed with the judge on all 19 labelled.
    "claude": Profile("claude-code", _claude_code, validated=True,
                      infra_failure=_claude_code_failure),
}
GENERIC = Profile("generic", _generic, validated=False)


LAUNCHERS = re.compile(r"^(python[\d.]*|node|npx|uv|uvx|bun|deno|pipx)$")


def agent_name(template: list[str]) -> str:
    """The agent's executable, as the round recorded it: `/opt/homebrew/bin/goose` -> goose.
    Behind a launcher (`python agent.py`, `npx some-agent`) it is the first non-flag
    argument instead, since naming every Python agent `python` merges unrelated agents."""
    if not template:
        return "unknown"
    name = re.sub(r"(\.exe|\.cmd)$", "", Path(template[0]).name.lower())
    if LAUNCHERS.match(name):
        rest = [a for a in template[1:] if not a.startswith("-") and a not in ("run", "-m")]
        if rest:
            return re.sub(r"\.(py|js|ts|mjs)$", "", Path(rest[0]).name.lower())
    return name


def profile_for(template: list[str]) -> Profile:
    return PROFILES.get(agent_name(template), GENERIC)


def cost_usd(stdout: str) -> float | None:
    """What the agent says it spent, when it says so (Claude Code's JSON does)."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("total_cost_usd"), (int, float)):
            return float(doc["total_cost_usd"])
    return None
