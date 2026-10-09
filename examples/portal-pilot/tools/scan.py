"""Count the rows in every section of a record. Read only; the verification tour."""

import argparse

import portal

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--record", required=True)
args = p.parse_args()
state = portal.load()
account = portal.need_session(state)
if account:
    rec = portal.record(state, account, args.record)
    portal.save(state)
    if rec is None:
        portal.out({"result": "error", "error": f"no record {args.record} in {account}"})
    else:
        portal.out({"result": "ok", "account": account, "record": args.record,
                    "counts": {s: len(rec["sections"].get(s, [])) for s in portal.SECTIONS}})
