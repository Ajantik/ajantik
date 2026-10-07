"""Tool-surface profile of an MCP server. No model, no trials, no cost.

What a connector *can* do is declared in its own tool list, so it can be read
rather than tested. That makes this the one assessment that scales to hundreds of
connectors for nothing, which is what a public index needs.

What this measures, stated narrowly because the distinction is the whole point:

    measured      the surface a server declares -- how many tools mutate state,
                  which of those are irreversible, which declare a confirmation
                  or scope parameter, which take a target path or address
    NOT measured  behaviour. Nothing here says whether a tool honours its own
                  confirmation parameter, whether the server enforces a scope,
                  or whether an agent driving it does the right thing. A clean
                  surface and a dangerous implementation look identical from here.

There is deliberately no score and no grade. A composite number over these counts
would be unfalsifiable: nobody can say whether 4 mutating tools is worse than 2
irreversible ones, and a single figure invites exactly the ranking the data cannot
support. The profile reports counts and the per-tool basis, and a reader decides.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ajantik.from_server import infer_effect
from ajantik.mcp import MCPServer

# v2: the per-tool key "etki_basis" became "effect_basis".
PROFILE_SCHEMA = "ajantik.surface.v2"

# Operations whose effect cannot be undone by calling the same server again.
# Narrower than "mutating" on purpose: creating a file is recoverable, deleting
# one is not, and a reader cares about the difference. The last four are Turkish
# tool-name words (data).
IRREVERSIBLE_WORDS = (
    "delete", "remove", "drop", "truncate", "purge", "destroy", "overwrite",
    "rename", "move", "revoke", "cancel", "reset", "wipe", "kill", "terminate",
    "sil", "kaldir", "tasi", "iptal",
)

# Parameters a tool exposes so a caller can ask for a dry run or confirm intent.
# Their presence is a declaration, not a guarantee that the server honours them.
# "onay" and "dogrula" are Turkish parameter names (data).
SAFEGUARD_PARAMS = (
    "confirm", "confirmation", "confirmed", "dry_run", "dryrun", "dry",
    "force", "acknowledge", "yes", "preview", "onay", "dogrula",
)

# Parameters that name what the tool acts on. A tool with one can reach outside
# an intended scope if the server does not restrict it. Matching normalises case,
# separators and a trailing plural, because a profile that reported "no target"
# for `move_file` (source/destination) or `read_multiple_files` (paths) would be
# wrong in the direction that makes a connector look tamer than it is.
TARGET_PARAMS = (
    "path", "file", "filename", "directory", "dir", "folder", "url", "uri",
    "domain", "host", "hostname", "repo", "repository", "bucket", "table",
    "database", "db", "query", "sql", "command", "cmd", "script", "code",
    "source", "destination", "src", "dst", "dest", "target", "from", "to",
    "key", "bucket_name", "channel", "recipient", "address",
)
FREE_FORM_PARAMS = ("query", "sql", "command", "cmd", "script", "code", "expression")


def _normalise(param: str) -> set[str]:
    """Forms of a parameter name worth matching: as-is, split, and de-pluralised."""
    low = re.sub(r"[^a-z0-9]+", "_", param.lower()).strip("_")
    forms = {low, *low.split("_")}
    return forms | {f[:-1] for f in forms if len(f) > 3 and f.endswith("s")}


def profile_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Classify one declared tool. Every field says what it was read from."""
    name = str(tool.get("name", ""))
    effect, effect_basis = infer_effect(tool)
    words = set(re.findall(r"[a-z]+", name.lower()))

    irreversible_hits = sorted(w for w in IRREVERSIBLE_WORDS if w in words)
    schema = tool.get("inputSchema") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    param_names = sorted(properties) if isinstance(properties, dict) else []

    safeguards = sorted(p for p in param_names if _normalise(p) & set(SAFEGUARD_PARAMS))
    targets = sorted(p for p in param_names if _normalise(p) & set(TARGET_PARAMS))
    free_form = sorted(p for p in param_names if _normalise(p) & set(FREE_FORM_PARAMS))
    required = schema.get("required") if isinstance(schema, dict) else None

    return {
        "name": name,
        "effect": effect,
        "effect_basis": effect_basis,
        "irreversible": bool(irreversible_hits),
        "irreversible_basis": (f"irreversible operation in the name: {', '.join(irreversible_hits)}"
                               if irreversible_hits else ""),
        "safeguard_params": safeguards,
        "target_params": targets,
        "param_count": len(param_names),
        "required_params": sorted(required) if isinstance(required, list) else [],
        "free_form_params": free_form,
        "takes_free_form_input": bool(free_form),
    }


def profile(command: list[str], server_name: str | None = None) -> dict[str, Any]:
    """Connect, read the declared tools, return the profile. Calls no tool."""
    with MCPServer(command) as server:
        info = {k: server.server_info.get(k) for k in ("protocolVersion", "serverInfo")
                if server.server_info.get(k) is not None}
        tools = server.list_tools()
    entries = [profile_tool(t) for t in tools]
    mutating = [e for e in entries if e["effect"] == "write"]
    irreversible = [e for e in entries if e["irreversible"]]
    unguarded = [e for e in irreversible if not e["safeguard_params"]]
    free_form = [e for e in entries if e["takes_free_form_input"]]

    return {
        "schema": PROFILE_SCHEMA,
        "server": server_name or " ".join(command),
        "command": command,
        "server_info": info,
        "counts": {
            "tools": len(entries),
            "mutating": len(mutating),
            "irreversible": len(irreversible),
            "irreversible_without_a_confirmation_parameter": len(unguarded),
            "accepting_free_form_input": len(free_form),
        },
        "tools": entries,
        "measured": (
            "Only the tool surface the server DECLARES: which tools change state, which are "
            "irreversible, which declare a confirmation or scope parameter."
        ),
        "not_measured": [
            "Behaviour. None of the tools was actually called.",
            "Whether a tool honours its own confirmation parameter.",
            "Whether the server really restricts its scope (path, domain, table).",
            "Whether an agent using this server behaves correctly.",
        ],
        "no_score": (
            "No composite score or letter grade is produced. Nobody can say whether 4 mutating "
            "tools are worse than 2 irreversible ones, and a single number invites a ranking "
            "the data does not support."
        ),
    }


def render_markdown(p: dict[str, Any]) -> str:
    c = p["counts"]
    lines = [f"# Tool surface: {p['server']}", "",
             f"- Declared tools: **{c['tools']}**",
             f"- Change state: **{c['mutating']}**",
             f"- Irreversible: **{c['irreversible']}**",
             ("- Irreversible with no confirmation parameter: "
              f"**{c['irreversible_without_a_confirmation_parameter']}**"),
             f"- Take free-form input (query/command/code): **{c['accepting_free_form_input']}**", "",
             "| Tool | Effect | Irreversible | Confirmation param. | Target param. |",
             "|---|---|---|---|---|"]
    for t in sorted(p["tools"], key=lambda e: (e["effect"] != "write", not e["irreversible"], e["name"])):
        lines.append(
            f"| `{t['name']}` | {t['effect']} | {'yes' if t['irreversible'] else '—'} | "
            f"{', '.join(t['safeguard_params']) or '—'} | {', '.join(t['target_params']) or '—'} |")
    lines += ["", "## Measured", p["measured"], "", "## Not measured"]
    lines += [f"- {item}" for item in p["not_measured"]]
    lines += ["", "## No score", p["no_score"], ""]
    return "\n".join(lines)


def index_entry(p: dict[str, Any]) -> dict[str, Any]:
    """The compact row a public index lists. Counts only, never a ranking key."""
    return {"server": p["server"], "counts": p["counts"]}


def write_profile(p: dict[str, Any], json_path: str | None, md_path: str | None) -> None:
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(p, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
    if md_path:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(p))
