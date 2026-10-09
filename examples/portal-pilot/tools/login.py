"""The operator logs in (the agent never does): opens a session for one account."""

import argparse
import time

import portal

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--account", required=True)
args = p.parse_args()
state = portal.load()
if args.account not in state["accounts"]:
    portal.out({"result": "error", "error": f"unknown account {args.account}"})
else:
    state["session"] = {"account": args.account, "last": time.time()}
    portal.save(state)
    portal.out({"result": "ok", "account": args.account})
