"""The command shim: scripts the agent runs in its shell, recorded without being changed."""

import json
import os
import subprocess
from pathlib import Path

from ajantik.shim import install, matched_script

PILOT = Path(__file__).parent.parent / "examples" / "portal-pilot"


def test_only_configured_scripts_are_ours(tmp_path):
    assert matched_script(["tools/fill.py", "--record", "R-1"], ["tools/*.py"], PILOT) == \
        "tools/fill.py"
    assert matched_script(["-u", "./tools/scan.py"], ["tools/*.py"], PILOT) == "tools/scan.py"
    assert matched_script([str(PILOT / "tools" / "scan.py")], ["tools/*.py"], PILOT) == \
        "tools/scan.py"
    assert matched_script(["-c", "print(1)"], ["tools/*.py"], PILOT) is None
    assert matched_script(["other.py"], ["tools/*.py"], PILOT) is None
    # from a subfolder, matched against the project root, not the cwd
    assert matched_script(["scan.py"], ["tools/*.py"], PILOT / "tools", root=PILOT) == \
        "tools/scan.py"


def test_record_mode_logs_the_call_and_changes_nothing(tmp_path):
    log = tmp_path / "calls.jsonl"
    config = install(tmp_path / "shim", ["python3"], ["tools/*.py"], "record", log)
    env = {"PATH": f"{tmp_path / 'shim'}{os.pathsep}{os.environ['PATH']}",
           "AJANTIK_SHIM": str(config), "PORTAL_HOME": str(tmp_path / "portal"),
           "HOME": os.environ.get("HOME", "")}

    def sh(cmd):
        return subprocess.run(["/bin/sh", "-c", cmd], cwd=PILOT, env=env, capture_output=True,
                              text=True, check=True).stdout

    assert json.loads(sh("python3 tools/login.py --account ACME"))["account"] == "ACME"
    status = json.loads(sh("python3 tools/status.py"))
    assert status == {"result": "ok", "account": "ACME"}
    assert sh('python3 -c "print(41 + 1)"').strip() == "42"  # not ours: untouched, unlogged
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["script"] for r in rows] == ["tools/login.py", "tools/status.py"]
    assert json.loads(rows[1]["stdout"]) == status and rows[1]["exit"] == 0
