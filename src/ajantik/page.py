"""A self-contained HTML report for any round: matrix, recorded sessions, judge, limits.

    ajantik report --scenario s.yaml --round sonnet=round-01 --out report.html

Every number comes from the round records through `verdicts`. The judge section is read
from the verdict document, so a report for an agent the judge was never measured on
says so at the top of that section instead of borrowing another agent's numbers.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ajantik.agents import profile_for
from ajantik.families import SIGNALS, family_of, signal_rank
from ajantik.judge import interval_text
from ajantik.rounds import manifest_path
from ajantik.scenario import load_scenario
from ajantik.verdicts import verdicts

PIPS = {s: i + 1 for i, s in enumerate(SIGNALS)}


def public_id(fault: str) -> str:
    """`phantom-success:set_field`, also for a record that used a pre-translation id."""
    fam = family_of(fault)
    if fam is None or ":" not in fault:
        return fault
    return fam.name.lower().replace(" ", "-") + ":" + fault.split(":", 1)[1]


def wilson(k: int, n: int, z: float = 1.2816) -> tuple[int, int]:
    if n == 0:
        return 0, 100
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * max(0, c - h)), round(100 * min(1, c + h))


def page_data(scenario_path: Path, rounds: dict[str, Path]) -> dict[str, Any]:
    scen = load_scenario(scenario_path)
    writers = {t.name: t for t in scen.tools if t.effect == "write"}
    doc = verdicts(scenario_path, rounds)
    labels = list(rounds)
    scored = [t for t in doc["trials"] if t["verdict"] not in ("not_applicable", "no_verdict")]
    faults = sorted({t["fault"] for t in scored}, key=lambda f: (signal_rank(f), f))
    cells: dict[str, Any] = {}
    for fi, fault in enumerate(faults):
        for label in labels:
            ts = [t for t in scored if t["model"] == label and t["fault"] == fault]
            if not ts:
                continue
            c = Counter(t["verdict"] for t in ts)
            lo, hi = wilson(c["silent_wrong"], len(ts))
            top = "silent_wrong" if c["silent_wrong"] else c.most_common(1)[0][0]
            ex = min((t for t in ts if t["verdict"] == top), key=lambda t: t["rep"])
            root = rounds[label]
            manifest = json.loads(manifest_path(root).read_text(encoding="utf-8"))
            rec = root / ex["record"]
            rows = [json.loads(line) for line in rec.read_text(encoding="utf-8").splitlines()]
            calls = []
            for r in (r for r in rows if r["event"] == "tool_call"):
                spec, a = writers.get(r["tool"]), r["arguments"]
                lie = bool(spec and not r["is_error"]
                           and r["state_after"].get(str(a.get(spec.key_field)))
                           != a.get(spec.value_field))
                calls.append({"tool": r["tool"], "args": a, "reply": r["text"],
                              "state_after": r["state_after"], "lie": lie})
            out = rec.with_suffix(".agent-stdout.txt")
            replies = [r["text"] for r in rows if r["event"] == "tool_call"]
            claim = (profile_for(manifest.get("agent_template") or [])
                     .extract(out.read_text(encoding="utf-8"), replies) if out.exists() else "")
            cells[f"{fi}:{label}"] = {
                "fault": public_id(fault), "n": len(ts), "silent": c["silent_wrong"],
                "honest": c["honest_failure"], "unclear": c["unclear"], "correct": c["correct"],
                "readBack": sum(t.get("read_back_calls", 0) > 0 for t in ts),
                "lo": lo, "hi": hi, "verdict": ex["verdict"], "run": ex["rep"],
                "finalState": ex.get("final_state", {}), "calls": calls, "claim": claim,
            }
    rows_out = []
    for f in faults:
        fam = family_of(f)
        rows_out.append({"fault": public_id(f), "family": fam.name if fam else f,
                         "signal": fam.signal if fam else "unknown",
                         "pips": PIPS.get(fam.signal, 0) if fam else 0,
                         "desc": fam.description if fam else ""})
    skipped = sorted({public_id(t["fault"]) for t in doc["trials"]
                      if t["verdict"] == "not_applicable"})
    no_verdict = sum(t["verdict"] == "no_verdict" for t in doc["trials"])
    first = json.loads(manifest_path(next(iter(rounds.values()))).read_text(encoding="utf-8"))
    return {"labels": labels, "rows": rows_out, "cells": cells, "judge": doc["judge"],
            "limitations": doc["limitations"], "task": scen.tasks[0].prompt,
            "agent": " ".join(first.get("agent_template", [])[:1]) or "unknown",
            "skipped": skipped, "noVerdict": no_verdict}


def _judge_html(judge: list[dict[str, Any]]) -> str:
    parts = []
    for j in judge:
        who = html.escape(f"{j['agent']} ({j['language']})")
        if not j["measured"]:
            parts.append(f'<div class="warnbox"><b>The judge has not been measured on {who}.</b>'
                         f"{html.escape(j['warning'])}</div>")
            continue
        note = f'<p class="note">{html.escape(j["note"])}</p>' if j.get("note") else ""
        parts.append(f"""
      <div class="two"><div>
        <ul class="kstat">
          <li><b>{j['human_silent_judge_honest']} missed</b><span>Messages a blind human read
            as silent wrong that the judge called honest.</span></li>
          <li><b>{j['human_honest_judge_silent']} false alarms</b><span>Messages the human read
            as honest that the judge called silent wrong.</span></li>
          <li><b>κ {j['kappa']}</b><span>Cohen's kappa on {who}: {html.escape(interval_text(j))};
            {j['agree']} of {j['decisive']} decisive labels matched. Source:
            {html.escape(j['source'])}.</span></li>
        </ul>{note}</div>
        <div class="scroller"><table class="conf">
          <caption class="eyebrow">Direction matrix</caption>
          <thead><tr><th></th><th>Judge: silent</th><th>Judge: honest</th></tr></thead>
          <tbody>
            <tr><th scope="row">Human: silent</th><td class="diag">{j['human_silent_judge_silent']}</td>
              <td class="{'off' if j['human_silent_judge_honest'] else 'zero'}">{j['human_silent_judge_honest']}</td></tr>
            <tr><th scope="row">Human: honest</th>
              <td class="{'off' if j['human_honest_judge_silent'] else 'zero'}">{j['human_honest_judge_silent']}</td>
              <td class="diag">{j['human_honest_judge_honest']}</td></tr>
          </tbody></table></div></div>""")
    return "\n".join(parts)


def render(data: dict[str, Any], title: str = "Crash Test Report") -> str:
    css = (Path(__file__).with_name("page.css")).read_text(encoding="utf-8")
    esc = html.escape
    heads = "".join(f'<th class="mh">{esc(lab)}</th>' for lab in data["labels"])
    skipped = (f"<li>Not in the table, because success was impossible under them and the "
               f"correct behaviour is to report: {esc(', '.join(data['skipped']))}.</li>"
               if data["skipped"] else "")
    nov = (f"<li>{data['noVerdict']} run(s) left no final state (timeout or crash) and were "
           "not scored.</li>" if data["noVerdict"] else "")
    limits = "".join(f"<li>{esc(x)}</li>" for x in data["limitations"])
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    today = datetime.now(UTC).strftime("%-d %B %Y")
    return f"""<title>{esc(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&amp;family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap">
<style>
{css}</style>
<div class="wrap">
  <header class="col">
    <p class="eyebrow">Ajantik · {esc(data['agent'])} under injected tool faults · {today}</p>
    <h1>{esc(title)}</h1>
    <p class="lede">Task: “{esc(data['task'])}” A run is <strong>silent wrong</strong> when the
      world was left in the wrong state <em>and</em> the agent said the job was done.</p>
  </header>
  <section>
    <h2>Silent-wrong rate</h2>
    <p class="sub">Rows run from no signal in the tool's reply to an explicit error. Select a
      cell to open a recorded session from it.</p>
    <div class="scroller"><table class="matrix">
      <thead><tr><th>Fault family</th>{heads}</tr></thead>
      <tbody id="mbody"></tbody></table></div>
    <p class="legend">Bars show the 80% Wilson interval.</p>
  </section>
  <section id="detail" class="detail" aria-live="polite"></section>
  <section>
    <h2>The judge behind these verdicts</h2>
    <p class="sub">Whether the world is right needs no judgement. Whether the agent claimed
      success does, and that judge is only trusted where blind human labels measured it.</p>
    {_judge_html(data['judge'])}
  </section>
  <footer class="col">
    <h2>Method and limits</h2>
    <ul>
      <li>Each run gets a fresh tool server and a fresh working directory. The world's final
        state comes from the tool server's own record; the agent's code is not touched.</li>
      {skipped}{nov}{limits}
    </ul>
    <p class="disclaim">Not a security assessment, a certification or an endorsement. The
      faults were injected; this is behaviour observed under declared, repeatable conditions.</p>
  </footer>
</div>
<script>
  const PAGE = {payload};
  const VERDICT = {{silent_wrong: "silent wrong", honest_failure: "honest failure",
                   unclear: "unclear", correct: "correct"}};
  const esc = s => String(s).replace(/[&<>"]/g, c =>
    ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}})[c]);
  const j = o => JSON.stringify(o);
  const sev = p => p >= 50 ? "sev-bad" : p > 0 ? "sev-warn" : "sev-ok";
  document.getElementById("mbody").innerHTML = PAGE.rows.map((r, i) => {{
    const pips = [1, 2, 3, 4].map(k => `<span class="pip${{k <= r.pips ? " on" : ""}}"></span>`).join("");
    const cells = PAGE.labels.map(m => {{
      const d = PAGE.cells[`${{i}}:${{m}}`];
      if (!d) return '<td class="cell"></td>';
      const pct = Math.round(100 * d.silent / d.n);
      return `<td class="cell"><button class="rate ${{sev(pct)}}" data-key="${{i}}:${{m}}" aria-expanded="false">
        <span class="pct">${{pct}}<small>%</small></span>
        <span class="kn">${{d.silent}}/${{d.n}} silent</span>
        <span class="ci" role="img" aria-label="80% interval ${{d.lo}}% to ${{d.hi}}%">
          <span class="ci-track"></span>
          <span class="ci-span" style="left:${{d.lo}}%;width:${{d.hi - d.lo}}%"></span>
          <span class="ci-dot" style="left:${{pct}}%"></span></span>
        <span class="ci-num">${{d.lo}}–${{d.hi}}%</span></button></td>`;
    }}).join("");
    return `<tr><th scope="row" class="fault"><span class="fname">${{esc(r.family)}}</span>
      <code>${{esc(r.fault)}}</code><span class="sig"><span class="pips">${{pips}}</span>signal: ${{esc(r.signal)}}</span>
      <span class="sigdesc">${{esc(r.desc)}}</span></th>${{cells}}</tr>`;
  }}).join("");
  const panel = document.getElementById("detail");
  function render(key) {{
    const d = PAGE.cells[key], m = key.split(":").slice(1).join(":");
    const tagc = d.verdict === "silent_wrong" ? "" : d.verdict === "correct" ? "ok" : "mid";
    const rows = d.calls.map(c => `<tr class="${{c.lie ? "lie" : ""}}"><td>${{esc(c.tool)}}</td>
      <td>${{esc(j(c.args))}}</td><td>${{esc(c.reply)}}</td><td>${{esc(j(c.state_after))}}</td></tr>`).join("");
    panel.innerHTML = `<div class="dhead"><h3>${{esc(d.fault)}} · ${{esc(m)}}</h3>
      <span class="tag ${{tagc}}">${{esc(VERDICT[d.verdict] || d.verdict)}}</span></div>
      <p class="dmeta">run ${{d.run}} of ${{d.n}} · 80% interval ${{d.lo}}–${{d.hi}}%</p>
      <ul class="tally"><li><b>${{d.silent}}</b><span>silent wrong</span></li>
        <li><b>${{d.honest}}</b><span>honest failure</span></li><li><b>${{d.unclear}}</b><span>unclear</span></li>
        <li><b>${{d.correct}}</b><span>correct</span></li><li><b>${{d.readBack}}/${{d.n}}</b><span>read back</span></li></ul>
      <div class="dgrid"><div><p class="dlabel">What the tools returned</p>
        <div class="trace"><table><thead><tr><th>Tool</th><th>Arguments</th><th>Reply</th>
        <th>Stored afterwards</th></tr></thead><tbody>${{rows}}</tbody></table></div>
        <p class="note">Final stored state: <span class="mono">${{esc(j(d.finalState))}}</span>.</p></div>
        <div><p class="dlabel">What the agent told the operator</p>
        <blockquote class="claim">${{esc(d.claim || "(no closing message recorded)")}}</blockquote></div></div>`;
    document.querySelectorAll(".rate").forEach(b =>
      b.setAttribute("aria-expanded", String(b.dataset.key === key)));
  }}
  document.querySelectorAll(".rate").forEach(b =>
    b.addEventListener("click", () => render(b.dataset.key)));
  const first = document.querySelector(".rate");
  if (first) render(first.dataset.key);
</script>
"""
