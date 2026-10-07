"""Generate a scenario from a real MCP server's tool schemas.

Writing a scenario by hand is the slowest part of testing an agent, and for a
third-party connector it is also guesswork. The server already publishes its tool
names, descriptions and input schemas, so that part can be read instead of written.

What is read and what is guessed, kept apart on purpose:

  read     tool name, description, input schema -- straight from `tools/list`
  guessed  `effect` (does the tool write?), from the name and description
  yours    the task, the checks, the output contract -- nobody can infer what a
           correct answer looks like for your job

The guess matters, because `effect` decides which fault modules apply: a phantom
save is only generated for a writing tool. A wrong guess therefore silently
changes what gets tested, so every inference is written into the file as a comment
and printed when the file is generated. Review it before spending anything.

`effect` is only ever `write` or `none` here, never `read`. In a scenario's own simulated
bench `read` means "return what a `write` tool stored", which is a property of that
bench, not of a real server; labelling a real read tool `read` would import a
meaning that does not hold.

Prefer `ajantik record`: it also captures real replies and the state checks.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ajantik.faults import MODULES
from ajantik.mcp import MCPError, MCPServer

MODULE_ORDER = tuple(MODULES)

# Verbs that indicate a tool changes something. English and Turkish, so tools named in
# either language are recognised; these are data matched against tool names. Deliberately
# broad: calling a read tool a writer loses a few fault modules, while calling a writer a
# reader would let the harness treat a mutating call as safe to probe.
WRITE_WORDS = (
    "write", "create", "update", "insert", "save", "store", "put", "post", "set", "add",
    "append", "delete", "remove", "move", "rename", "edit", "patch", "upload", "send",
    "commit", "push", "execute", "run", "yaz", "kaydet", "olustur", "guncelle", "sil",
    "ekle", "gonder",
)
READ_WORDS = (
    "read", "get", "list", "search", "query", "fetch", "show", "describe", "find", "info",
    "stat", "view", "oku", "getir", "listele", "ara", "goster",
)


def infer_effect(tool: dict[str, Any]) -> tuple[str, str]:
    """Return (effect, basis). `basis` is the evidence that decided it, for the comment.

    The name is the strongest signal and is settled first, both ways. A description
    is weaker evidence because descriptions routinely mention the other operations a
    tool relates to -- "read the whole store" names a store without writing to one,
    and letting that outvote a name like `read_all` turns a read tool into a write tool and
    generates phantom-save conditions that cannot mean anything.
    """
    name = str(tool.get("name", ""))
    words = set(re.findall(r"[a-z]+", name.lower()))
    hits = sorted(w for w in WRITE_WORDS if w in words)
    if hits:
        return "write", f"write verb in the name: {', '.join(hits)}"
    read_hits = sorted(w for w in READ_WORDS if w in words)
    if read_hits:
        return "none", f"read verb in the name: {', '.join(read_hits)}; description not consulted"
    desc = str(tool.get("description", "")).lower()
    desc_hits = sorted({w for w in WRITE_WORDS if re.search(rf"\b{w}", desc)})
    if desc_hits:
        return "write", f"write verb in the description: {', '.join(desc_hits)}"
    return "none", "no write verb found (default: assumed not to change anything)"


def sample_responses(server: MCPServer, tools: list[dict[str, Any]],
                     effects: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Call each NON-writing tool once with empty arguments to capture a sample.

    Returns (samples, skipped) where `skipped` says why each tool produced none.
    Nothing here fails quietly: a tool that could not be sampled is reported, so
    a missing `response` in the generated file is explained rather than looking
    like the tool returns nothing.

    Opt-in, and never for a tool inferred as writing: a sample is worth having
    (it is what `data_anomaly` derives metamorphic anomalies from) but not at the
    price of a side effect nobody asked for.
    """
    out: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for tool in tools:
        name = tool["name"]
        if effects.get(name) != "none":
            skipped[name] = "inferred as writing; not called, to avoid a side effect"
            continue
        try:
            result = server.call_tool(name, {})
        except (MCPError, OSError) as exc:
            skipped[name] = f"call failed: {exc}"
            continue
        if result.get("isError"):
            skipped[name] = "the tool returned an error (called with empty arguments)"
            continue
        text = "\n".join(
            item["text"] for item in (result.get("content") or [])
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
        if text:
            out[name] = text
        else:
            skipped[name] = "returned no text content"
    return out, skipped


def scenario_yaml(command: list[str], tools: list[dict[str, Any]], skill: str,
                  task_prompt: str, samples: dict[str, str] | None = None) -> str:
    """Render the scenario file. Inferences are commented, not hidden."""
    samples = samples or {}
    effects: dict[str, tuple[str, str]] = {t["name"]: infer_effect(t) for t in tools}
    lines = [
        "# Generated by `ajantik scenario-from-server`. REVIEW BEFORE SPENDING.",
        "#",
        "# READ from the server: tool names, descriptions, input schemas.",
        "# INFERRED: each tool's `effect` (does it write?). It decides which fault modules",
        "#   apply; if it is wrong, you silently measure something else.",
        "# YOURS: the task, the checks, the output contract. No schema knows what a",
        "#   correct answer looks like.",
        "",
        "mcp_server:",
    ]
    lines += [f"- {_q(part)}" for part in command]
    lines += ["", f"skill: {skill}", "", "tools:"]
    for tool in tools:
        name = tool["name"]
        effect, basis = effects[name]
        lines.append(f"# effect inferred: {basis}")
        lines.append(f"- name: {_q(name)}")
        lines.append(f"  description: {_q(str(tool.get('description', '')).strip())}")
        lines.append(f"  effect: {effect}")
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        lines.append(f"  input_schema: {json.dumps(schema, ensure_ascii=False)}")
        if name in samples:
            lines.append(f"  response: {_q(samples[name])}")
    lines += [
        "",
        "tasks:",
        "- id: task-1",
        f"  prompt: {_q(task_prompt)}",
        "  checks: []            # FILL IN: checks that tell a right answer from a wrong one",
        "",
        "# Faults are generated from modules. To see which module applies to which tool:",
        "# ajantik faults <this file>",
        "auto_faults:",
    ]
    # `data_anomaly` derives metamorphic anomalies from a tool's sample reply. Without one it
    # would derive them from the loader's placeholder ('{"status": "ok"}') and produce
    # conditions unrelated to the real tool -- measuring noise. So the module is listed only
    # when a real sample was captured.
    lines += [f"- {name}" for name in MODULE_ORDER if name != "data_anomaly" or samples]
    if not samples:
        lines += [
            "# data_anomaly not listed: no sample reply. Anomalies derived from a placeholder",
            "# do not measure the real tool. Capture samples with --sample-replies, or write",
            "# a 'response:' for the tool by hand and add the module.",
        ]
    lines += [
        "",
        "faults:",
        "- id: clean",
        "",
        "# output_contract: ...   # fill in if a program reads the output; that is where",
        "#                        # silent failures really show",
    ]
    return "\n".join(lines) + "\n"


def generate(command: list[str], skill: str, task_prompt: str,
             with_samples: bool = False) -> tuple[str, dict[str, tuple[str, str]], dict[str, str]]:
    """Connect, read the tools, return (yaml_text, inferences, skipped_samples).

    The caller prints the inferences: a guessed `effect` decides which faults get
    generated, so it must be reviewed by a person before anything is spent.
    """
    with MCPServer(command) as server:
        tools = server.list_tools()
        if not tools:
            raise ValueError("The server declared no tools (tools/list is empty).")
        effects = {t["name"]: infer_effect(t) for t in tools}
        samples: dict[str, str] = {}
        skipped: dict[str, str] = {}
        if with_samples:
            samples, skipped = sample_responses(
                server, tools, {k: v[0] for k, v in effects.items()})
    return scenario_yaml(command, tools, skill, task_prompt, samples), effects, skipped


def _q(text: str) -> str:
    """Always quote: tool descriptions contain colons, hashes and newlines."""
    return json.dumps(text.replace("\n", " ").strip(), ensure_ascii=False)
