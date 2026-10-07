"""Track record report (Markdown) and badge (SVG) from all recorded trials of one identity."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ajantik import legacy
from ajantik.estimate import CostSummary, RateSummary, summarize_cost, summarize_rate
from ajantik.identity import Identity
from ajantik.pricing import PRICE_SOURCE

NOT_MEASURED = "not measured"




def load_trials(path: Path) -> list[dict]:
    """Read one identity's trial records, in the current vocabulary.

    The single entry point for trial records. Records written before the English
    vocabulary use other fault ids, claim labels and field names; they are mapped here
    so every reader sees one form.
    """
    if not path.exists():
        return []
    from ajantik.faults import canonical_id
    from ajantik.oracle import canonical_label

    trials = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for t in trials:
        t["fault"] = canonical_id(t["fault"])
        if t.get("outcome"):
            t["outcome"] = canonical_label(t["outcome"])
        for old, new in legacy.TRIAL_FIELDS.items():
            # A record holds both keys only if an older writer updated it after it was
            # normalised; the older key then carries the newer value.
            if old in t:
                t[new] = t.pop(old)
    return trials


def _pct(x: float | None) -> str:
    return NOT_MEASURED if x is None else f"{100 * x:.0f}%"


def _usd(x: float | None, digits: int = 4) -> str:
    return NOT_MEASURED if x is None else f"${x:.{digits}f}"


def _rate_text(r: RateSummary) -> str:
    if r.n == 0:
        return NOT_MEASURED
    text = f"{r.successes}/{r.n} ({_pct(r.rate)}; 80% interval {_pct(r.low)}–{_pct(r.high)})"
    if r.max_failure_rate_95 is not None and r.max_failure_rate_95 < 1:
        text += (f"; no failure seen, failure rate at most {_pct(r.max_failure_rate_95)} "
                 "with 95% confidence")
    return text


def _cost_text(c: CostSummary) -> str:
    if c.n == 0:
        return NOT_MEASURED
    if c.p10 is None:
        return f"single measurement {_usd(c.p50)}, no interval can be computed"
    return f"typical {_usd(c.p50)} · p10–p90 {_usd(c.p10)}–{_usd(c.p90)} · mean {_usd(c.mean)}"


def build_report(identity: Identity, skill_name: str, trials: list[dict]) -> tuple[str, dict]:
    finished = [t for t in trials if t["success"] is not None]
    unfinished = [t for t in trials if t["success"] is None]
    by_fault: dict[str, list[dict]] = defaultdict(list)
    for t in finished:
        by_fault[t["fault"]].append(t)

    overall_rate = summarize_rate([t["success"] for t in finished])
    clean = by_fault.get("clean", [])
    clean_cost = summarize_cost([t["cost_usd"] for t in clean])
    spent = sum(t["cost_usd"] for t in trials)

    lines = [
        f"# Track record: {skill_name}",
        "",
        (
            f"Identity `{identity.id}` · recipe `{identity.recipe[:12]}` · model `{identity.model}` "
            f"· effort `{identity.effort}` · {identity.harness}"
        ),
        "",
        (
            "Every figure in this report was measured from the trials below. "
            f'Where there is no measurement it says "{NOT_MEASURED}".'
        ),
        "",
        (
            "**Right behaviour:** success when success is possible, otherwise saying plainly "
            "that it failed. **Silent wrong:** the result is wrong but the agent said it "
            "succeeded; the most dangerous case."
        ),
        "",
        "## Summary",
        "",
        f"- **Success (all conditions):** {_rate_text(overall_rate)}",
        f"- **Cost of one run (clean condition):** {_cost_text(clean_cost)}",
    ]
    if clean_cost.total_p50 is not None:
        lines.append(
            f"- **Total for {clean_cost.total_runs:,} runs (clean condition):** "
            f"typical {_usd(clean_cost.total_p50, 2)} · p10–p90 "
            f"{_usd(clean_cost.total_p10, 2)}–{_usd(clean_cost.total_p90, 2)}"
        )
    if clean_cost.widening > 1:
        lines.append(
            f"- With few measurements the band was widened {clean_cost.widening:.2f}× "
            "(the mean is not widened)."
        )
    lines += [
        "",
        "## By fault condition",
        "",
        (
            "| Condition | Success | Right behaviour | Silent wrong | Cost of one run "
            "| Mean turns | Cost vs clean |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    clean_mean = clean_cost.mean
    for fault in sorted(by_fault, key=lambda f: (f != "clean", f)):
        ts = by_fault[fault]
        rate = summarize_rate([t["success"] for t in ts])
        cost = summarize_cost([t["cost_usd"] for t in ts])
        turns = sum(t["turns"] for t in ts) / len(ts)
        ratio = f"{cost.mean / clean_mean:.2f}×" if clean_mean and cost.mean is not None else NOT_MEASURED
        judged = [t for t in ts if t.get("right") is not None]
        right = f"{sum(t['right'] for t in judged)}/{len(judged)}" if judged else NOT_MEASURED
        silent = sum(t.get("outcome") == "silent_wrong" for t in ts)
        silent_txt = (f"⚠️ {silent}" if silent else "0") if judged else NOT_MEASURED
        lines.append(
            f"| {fault} | {_rate_text(rate)} | {right} | {silent_txt} | {_cost_text(cost)} "
            f"| {turns:.1f} | {ratio} |"
        )

    failures = [t for t in finished if not t["success"]]
    if failures:
        lines += ["", "## Failed trials", ""]
        for t in failures:
            why = ", ".join(t["failed_checks"]) or f"stop reason: {t['stop']}"
            lines.append(f"- `{t['fault']}` #{t['rep']}: {why}")
    lines += [
        "",
        "## Record",
        "",
        f"- Finished trials: {len(finished)} · unfinished (not evidence): {len(unfinished)}",
        f"- Total spent on this identity: ${spent:.4f}",
        f"- Price source: {PRICE_SOURCE}",
    ]
    summary = {
        "identity": identity.to_dict(),
        "n": overall_rate.n,
        "success_rate": overall_rate.rate,
        "clean_cost_p50": clean_cost.p50,
    }
    return "\n".join(lines) + "\n", summary


def badge_svg(summary: dict) -> str:
    rate = summary["success_rate"]
    cost = summary["clean_cost_p50"]
    right = NOT_MEASURED if rate is None else f"{100 * rate:.0f}% · {summary['n']} trials"
    if cost is not None:
        right += f" · ${cost:.3f}"
    lw, rw = 70, 12 + 7 * len(right)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + rw}" height="20" role="img" '
        f'aria-label="Ajantik track record: {right}">'
        f'<rect width="{lw}" height="20" fill="#444441"/>'
        f'<rect x="{lw}" width="{rw}" height="20" fill="#0F6E56"/>'
        '<g fill="#fff" font-family="Verdana,sans-serif" font-size="11">'
        f'<text x="8" y="14">ajantik</text><text x="{lw + 6}" y="14">{right}</text></g></svg>'
    )


def build_comparison(before_id: str, after_id: str, before: list[dict], after: list[dict]) -> str:
    """Before/after table per fault condition; only finished trials count as evidence."""

    def group(trials: list[dict]) -> dict[str, list[dict]]:
        g: dict[str, list[dict]] = defaultdict(list)
        for t in trials:
            if t["success"] is not None:
                g[t["fault"]].append(t)
        return g

    ga, gb = group(before), group(after)
    faults = sorted(set(ga) | set(gb), key=lambda f: (f != "clean", f))
    lines = [
        f"# Before / after: `{before_id}` → `{after_id}`",
        "",
        (
            "| Condition | Success before | Success after | Right behaviour before "
            "| Right behaviour after | Silent wrong before | Silent wrong after "
            "| Mean cost before | Mean cost after |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in faults:
        ra = summarize_rate([t["success"] for t in ga.get(f, [])])
        rb = summarize_rate([t["success"] for t in gb.get(f, [])])
        ca = summarize_cost([t["cost_usd"] for t in ga.get(f, [])])
        cb = summarize_cost([t["cost_usd"] for t in gb.get(f, [])])
        sa = f"{ra.successes}/{ra.n}" if ra.n else NOT_MEASURED
        sb = f"{rb.successes}/{rb.n}" if rb.n else NOT_MEASURED
        def right(ts: list[dict]) -> str:
            judged = [t for t in ts if t.get("right") is not None]
            return f"{sum(t['right'] for t in judged)}/{len(judged)}" if judged else NOT_MEASURED

        def silent(ts: list[dict]) -> str:
            if not any(t.get("outcome") for t in ts):
                return NOT_MEASURED
            return str(sum(t.get("outcome") == "silent_wrong" for t in ts))

        lines.append(
            f"| {f} | {sa} | {sb} | {right(ga.get(f, []))} | {right(gb.get(f, []))} "
            f"| {silent(ga.get(f, []))} | {silent(gb.get(f, []))} "
            f"| {_usd(ca.mean)} | {_usd(cb.mean)} |"
        )
    ta = summarize_rate([t["success"] for t in before if t["success"] is not None])
    tb = summarize_rate([t["success"] for t in after if t["success"] is not None])
    lines += [
        "",
        f"- **Total success:** before {_rate_text(ta)} → after {_rate_text(tb)}",
    ]
    ra = summarize_rate([t["right"] for t in before if t.get("right") is not None])
    rb = summarize_rate([t["right"] for t in after if t.get("right") is not None])
    if ra.n and rb.n:
        lines.append(
            f"- **Total right behaviour:** before {_rate_text(ra)} → after {_rate_text(rb)}"
        )
    return "\n".join(lines) + "\n"
