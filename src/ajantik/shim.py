"""Command shim: the fault boundary for skills whose tools are scripts, not MCP servers.

Many serious skills reach the world through scripts the agent runs in its shell
(`python3 tools/fill.py ...`, `node runner/save.js ...`), each printing one JSON line. The
agent reasons over that line, so that line is where a fault belongs. The shim puts a small
launcher with the same name (`python3`, `node`, `bash`) first on the agent's PATH. A call
whose script matches one of the configured patterns is handled here; every other call goes
straight to the real program, untouched.

Modes, from the config named by AJANTIK_SHIM (a JSON file):

    pass    run the real script, record nothing
    record  run the real script, append (script, args, cwd, stdout, exit code, time) to a log
    twin    never run the real script: `ajantik.scripted` answers from a twin of the world,
            through a skill-specific adapter, applying the run's fault ("twin" and "adapter"
            in the config)

The config, written by `install`:

    {"mode": "record", "log": "/abs/calls.jsonl",
     "launchers": {"python3": "/usr/bin/python3"},       # the real programs
     "patterns": ["tools/*.py"]}                         # scripts to handle, relative to cwd
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _config() -> dict[str, Any]:
    path = os.environ.get("AJANTIK_SHIM")
    if not path:
        raise SystemExit("ajantik shim: AJANTIK_SHIM is not set")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matched_script(args: list[str], patterns: list[str], cwd: Path,
                   root: Path | None = None) -> str | None:
    """The script a launcher call runs, if it is one we handle: the first argument that is
    not a flag, as a path relative to the project root (default: cwd). Relative to the root,
    not to cwd, so `cd runner && node save.js` still matches `runner/*.js`."""
    script = next((a for a in args if not a.startswith("-")), None)
    if script is None:
        return None
    base = (root or cwd).resolve()
    full = (cwd / script).resolve()
    rel = str(full.relative_to(base)) if full.is_relative_to(base) else script
    rel = rel.removeprefix("./")
    return rel if any(fnmatch.fnmatch(rel, pat) for pat in patterns) else None


def run(launcher: str, args: list[str]) -> int:
    cfg = _config()
    real = cfg["launchers"][launcher]
    cwd = Path.cwd()
    root = Path(cfg["root"]) if cfg.get("root") else None
    script = matched_script(args, cfg.get("patterns", []), cwd, root)
    if script is None and cfg.get("mode") == "twin" and launcher in cfg.get("block", {}):
        # Fail closed: in a twin run, an unknown call of this launcher must not reach the
        # real system (a script the adapter does not know could drive a live session).
        sys.stdout.write(cfg["block"][launcher].rstrip("\n") + "\n")
        return 1
    if script is None or cfg.get("mode", "pass") == "pass":
        os.execv(real, [real, *args])  # not ours: the real program, as if we were not here
    if cfg["mode"] == "twin":
        from ajantik import scripted
        from ajantik.proxy import _lock_file, _unlock_file

        twin = Path(cfg["twin"])
        with open(twin.with_suffix(".lock"), "a+") as lock:
            _lock_file(lock)
            try:
                out, code = scripted.handle(twin, scripted.load_adapter(cfg["adapter"]),
                                            script, args)
            finally:
                _unlock_file(lock)
        sys.stdout.write(out)
        return code
    started = time.monotonic()
    done = subprocess.run([real, *args], capture_output=True, text=True, check=False)
    row = {"at": datetime.now(UTC).isoformat(timespec="milliseconds"), "launcher": launcher,
           "script": script, "args": args, "cwd": str(cwd), "stdout": done.stdout,
           "stderr": done.stderr[-2000:], "exit": done.returncode,
           "ms": round((time.monotonic() - started) * 1000)}
    with open(cfg["log"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode


def install(shim_dir: Path, launchers: list[str], patterns: list[str], mode: str,
            log: Path, path: str | None = None, **extra: Any) -> Path:
    """Write the launchers and the config. Returns the config path; put shim_dir first on the
    agent's PATH and AJANTIK_SHIM=<config> in its environment."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    search = os.pathsep.join(p for p in (path or os.environ.get("PATH", "")).split(os.pathsep)
                             if Path(p).resolve() != shim_dir.resolve())
    real = {}
    for name in launchers:
        found = shutil.which(name, path=search)
        if not found:
            raise ValueError(f"{name} is not on PATH")
        real[name] = str(Path(found).absolute())
        launcher = shim_dir / name
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m ajantik.shim {name} "$@"\n',
                            encoding="utf-8")
        launcher.chmod(0o755)
    config = shim_dir / "shim.json"
    config.write_text(json.dumps({"mode": mode, "log": str(log.absolute()), "launchers": real,
                                  "patterns": patterns, **extra}, indent=1), encoding="utf-8")
    return config


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m ajantik.shim <launcher> [args...]")
    return run(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
