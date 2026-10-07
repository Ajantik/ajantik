"""Call a guard function written in Python or JavaScript, in batches.

Targets:  py:path/to/file.py#function   or   js:path/to/file.js#exportName
A JS guard runs in one `node` process per batch; it never touches a browser or a network.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

JS_RUNNER = r"""
const fs = require('fs');
const { file, name, calls } = JSON.parse(fs.readFileSync(0, 'utf8'));
const mod = require(file);
const fn = mod[name];
if (typeof fn !== 'function') { console.error(`'${name}' is not a function`); process.exit(2); }
const out = calls.map(args => { try { return { ok: true, value: fn(...args) }; }
                                catch (e) { return { ok: false, error: String(e && e.message || e) }; } });
process.stdout.write(JSON.stringify(out));
"""


class Guard:
    def __init__(self, target: str, base: Path):
        kind, _, rest = target.partition(":")
        path_s, _, name = rest.partition("#")
        if kind not in ("py", "js") or not path_s or not name:
            raise ValueError(
                f"A guard target must be 'py:file.py#function' or 'js:file.js#function': {target}"
            )
        self.kind, self.name = kind, name
        self.path = (base / path_s).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._py: Callable[..., Any] | None = None
        if kind == "py":
            spec = importlib.util.spec_from_file_location(f"guard_{self.path.stem}", self.path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._py = getattr(module, name)
        elif not shutil.which("node"):
            raise RuntimeError("A JS guard needs 'node'.")

    def call_many(self, calls: list[list[Any]]) -> list[dict[str, Any]]:
        if self._py is not None:
            out = []
            for args in calls:
                try:
                    out.append({"ok": True, "value": self._py(*args)})
                except Exception as e:  # noqa: BLE001 — a crash on odd input is itself a finding
                    out.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
            return out
        payload = json.dumps({"file": str(self.path), "name": self.name, "calls": calls})
        proc = subprocess.run(
            ["node", "-e", JS_RUNNER], input=payload, capture_output=True, text=True, timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "node failed")
        return json.loads(proc.stdout)
