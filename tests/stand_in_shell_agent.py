"""A stand-in for `claude -p` on a script-driven skill: runs the pilot's tools through the shell
(so the shim answers them), blind, and prints Claude Code's JSON. No model call."""

import json
import subprocess

ROWS = {"1": [{"name": "x"}], "2": [{}, {}, {}], "3": [{}, {}], "4": [{}], "5": [{}, {}, {}, {}],
        "6": [{"text": "t"}], "8": [{"email": "e"}]}


def sh(cmd):
    return subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True,
                          check=False).stdout


sh("python3 tools/status.py")
sh("python3 tools/open_record.py --record R-102")
for section, rows in ROWS.items():
    sh(f"python3 tools/fill.py --record R-102 --section {section} --rows '{json.dumps(rows)}'")
sh("python3 tools/upload.py --record R-102 --file docs-in/R-102.pdf")
report = "R-102: VERIFIED 8/8"
print(json.dumps({"type": "result", "result": report, "num_turns": 3, "total_cost_usd": 0.01}))
