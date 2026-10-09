"""Attach a document to section 7 of a record. Writes. Stores whatever bytes it reads."""

import argparse
from pathlib import Path

import portal

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--record", required=True)
p.add_argument("--file", required=True)
args = p.parse_args()
state = portal.load()
account = portal.need_session(state)
if account:
    rec = portal.record(state, account, args.record)
    path = Path(args.file)
    if rec is None or not path.is_file():
        portal.save(state)
        portal.out({"result": "error",
                    "error": f"no record {args.record}" if rec is None else f"no file {path}"})
    else:
        size = path.stat().st_size
        rec["sections"].setdefault("7", []).append({"file": path.name, "bytes": size})
        portal.save(state)
        portal.out({"result": "ok", "account": account, "record": args.record, "section": "7",
                    "file": path.name, "bytes": size})
