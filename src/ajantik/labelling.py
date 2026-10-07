"""Turn round records into a labelling set, so the oracle can be measured on them.

The oracle only has a question to answer when the result was wrong: did the agent
report the problem, or present the wrong result as if it were right? So this keeps
exactly the trials where the world ended in the wrong state *and* success was
possible, pairs each with the agent's closing message, and records the oracle's own
label as the reference a human is checked against.

The closing message is extracted, not the whole transcript. Feeding the transcript
counts the tools' own "saved" echoes as the agent's claim, which flips the verdict.
This was measured, not assumed.

    python -m ajantik.labelling --scenario examples/intake-form/scenario.yaml \
        --round haiku=/tmp/round-haiku --round sonnet=/tmp/round-sonnet \
        --out /tmp/judge-set.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from ajantik import legacy
from ajantik.faults import canonical_id
from ajantik.oracle import CURRENT, classify
from ajantik.rounds import manifest_path as round_manifest
from ajantik.scenario import load_scenario

# Goose prints a banner before the run; the agent's own words start after it.
BANNER_END = "goose is ready"
SEPARATOR = re.compile(r"^\s*[\u2500-\u257f]+\s*$", re.MULTILINE)


def closing_message(stdout: str, tool_texts: list[str] | None = None) -> str:
    """The agent's closing prose, cut out by exact match rather than by guessing.

    Scraping a terminal is fragile: a tool that answers with plain text instead of
    JSON (`503 Service Unavailable`) defeats any pattern, and its reply gets read as
    the agent's own claim. But the tool server recorded every reply verbatim, so the
    cut can be exact. Everything after the last recorded reply is the agent's.
    """
    text = stdout
    if BANNER_END in text:
        text = text.split(BANNER_END, 1)[1]
    # The last separator opens the last tool call, so the agent's closing words are
    # inside that final segment. Cutting on the last occurrence of the reply anywhere
    # would land inside the agent's own prose, because agents quote the reply they got
    # ("...(503 Service Unavailable). Please..."). Within the final segment the echo
    # comes first, so cut there.
    segments = SEPARATOR.split(text)
    text = segments[-1] if segments else text
    for reply in reversed([r for r in (tool_texts or []) if r]):
        found = text.find(reply)
        if found >= 0:
            text = text[found + len(reply):]
            break
    return text.strip()


def collect(scenario_path: Path, rounds: dict[str, Path], oracle: int = CURRENT) -> list[dict]:
    scenario = load_scenario(scenario_path)
    possible = {f.id: f.success_possible for f in scenario.faults}
    declared = scenario.tasks[0].state_checks
    if not declared:
        raise SystemExit("The scenario's task declares no state_checks, so there is no way "
                         "to tell whether the world ended right.")
    from ajantik.agents import profile_for

    items = []
    for model, root in rounds.items():
        manifest_path = round_manifest(root)
        template = (json.loads(manifest_path.read_text(encoding="utf-8")).get("agent_template")
                    if manifest_path.exists() else None) or []
        extract = profile_for(template).extract if template else closing_message
        # Older lab rounds keep their records under another directory name (ajantik.legacy).
        records = [p for d in ("records", *legacy.RECORD_DIRS) for p in (root / d).glob("*.jsonl")]
        for record in sorted(records):
            rows = [json.loads(l) for l in record.read_text(encoding="utf-8").splitlines()]
            start = next((r for r in rows if r["event"] == "session_start"), None)
            end = next((r for r in rows if r["event"] == "session_end"), None)
            if start is None or end is None:
                continue  # cut short: no recorded final state, nothing to judge
            fault = canonical_id(start["fault"])
            if not possible.get(fault, True):
                continue  # success was not possible; reaching the state is not the goal
            if all(c.passes(end["final_state"]) for c in declared):
                continue  # the world is right; the oracle has no question
            out = record.with_suffix(".agent-stdout.txt")
            if not out.exists():
                continue
            replies = [r.get("text", "") for r in rows if r.get("event") == "tool_call"]
            text = extract(out.read_text(encoding="utf-8"), replies)
            items.append({
                "id": f"{model}/{record.stem}",
                "scenario": str(scenario_path),
                "label": classify(False, "end_turn", text,
                                   failure_words=scenario.failure_words,
                                   success_words=scenario.success_words,
                                   contract=scenario.contract,
                                   status_field=scenario.status_field, version=oracle),
                # `consumer` is the group the agreement report aggregates by;
                # `consumer_note` is the line the labeller reads on the form.
                "consumer": "human",
                "consumer_note": "a person (the operator reading the agent's closing message)",
                "reason": f"fault {fault}; final state "
                          f"{json.dumps(end['final_state'], ensure_ascii=False)}",
                "text": text,
            })
    return items


def sample(items: list[dict], size: int, seed: int = 20261006) -> list[dict]:
    """A stratified sample, because a round yields more cases than a person can label.

    Kappa needs both classes present, so the sample is balanced across the oracle's own
    labels rather than drawn at random; every abstention is kept, since those are the
    cases where the two labellers are most likely to part. That makes the set unfit for
    estimating how common each class is, and the report must say so.
    """
    if size >= len(items):
        return items
    import random

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it["label"], []).append(it)
    out = list(groups.pop("unclear", []))  # keep every abstention
    per = max(1, (size - len(out)) // max(1, len(groups)))
    for label in sorted(groups):
        pool = groups[label]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out[:size]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ajantik.labelling", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, type=Path, dest="scenario")
    p.add_argument("--round", action="append", required=True, dest="rounds",
                   help="model=directory (repeatable)")
    p.add_argument("--out", required=True, type=Path, dest="out")
    p.add_argument("--oracle", type=int, default=CURRENT, dest="oracle")
    p.add_argument("--sample", type=int, dest="sample",
                   help="stratified sample size (default: every case)")
    args = p.parse_args(argv)

    rounds = {}
    for spec in args.rounds:
        if "=" not in spec:
            raise SystemExit(f"--round expects 'model=directory', got '{spec}'")
        model, path = spec.split("=", 1)
        rounds[model] = Path(path)

    items = collect(args.scenario, rounds, args.oracle)
    total = len(items)
    if args.sample:
        items = sample(items, args.sample)
    if not items:
        raise SystemExit("No case to judge: every trial ended with the world right, or "
                         "success was impossible.")
    doc = {
        "description": ("Judge measurement set: closing messages from trials of an agent we do "
                        "not own (an external MCP client) that ended with the world wrong. The "
                        f"label was proposed by judge v{args.oracle} and is kept as the reference "
                        "an independent human label is measured against."),
        "labelled_by": f"judge v{args.oracle} (awaiting a human check)",
        "items": items,
    }
    if args.sample and len(items) < total:
        doc["description"] += (f" {len(items)} of {total} cases were sampled by stratum (fixed "
                               "seed, every abstention kept, classes balanced): fit for measuring "
                               "agreement, not for estimating how common each class is.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.dump(doc, allow_unicode=True, sort_keys=False, width=100),
                        encoding="utf-8")
    counts: dict[str, int] = {}
    for i in items:
        counts[i["label"]] = counts.get(i["label"], 0) + 1
    print(f"{len(items)} items -> {args.out}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
