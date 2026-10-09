"""Open a record of the logged-in account: which sections already have rows. Read only."""

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
                    "product": rec["product"],
                    "filled": sorted(s for s, rows in rec["sections"].items() if rows)})
