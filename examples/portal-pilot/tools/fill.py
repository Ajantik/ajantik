"""Save rows into one section of a record ("Save" once per row). Writes."""

import argparse
import json

import portal

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--record", required=True)
p.add_argument("--section", required=True, choices=sorted(portal.SECTIONS))
p.add_argument("--rows", required=True, help="JSON list of objects, one per row")
args = p.parse_args()
state = portal.load()
account = portal.need_session(state)
if account:
    rec = portal.record(state, account, args.record)
    try:
        rows = json.loads(args.rows)
        assert isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
    except (ValueError, AssertionError):
        rows = None
    if rec is None:
        portal.save(state)
        portal.out({"result": "error", "error": f"no record {args.record} in {account}"})
    elif rows is None:
        portal.save(state)
        portal.out({"result": "error", "error": "--rows must be a JSON list of objects"})
    else:
        rec["sections"].setdefault(args.section, []).extend(rows)
        portal.save(state)
        portal.out({"result": "ok", "account": account, "record": args.record,
                    "section": args.section, "saved": len(rows), "accepted": len(rows)})
