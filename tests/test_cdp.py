"""The CDP guard against a real, headless Chrome (skipped when Chrome is not installed)."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.client import connect

from ajantik.cdp import READY_PREFIX, Fault, Policy, rewrite

# -- the policy, without a browser -------------------------------------------------------------


def test_policy_blocks_writes_and_page_loads_not_reads():
    p = Policy.from_dict({"deny": ["/submit"], "allow_writes": [r"^https://app\.example/"]})
    assert p.blocks("POST", "https://app.example/submit/1") == "write to a denied URL"
    assert p.blocks("PUT", "https://other.example/x") == "write outside allow_writes"
    assert p.blocks("PUT", "https://app.example/doc/1") is None
    assert p.blocks("GET", "https://app.example/submit/1", "Document") == "page load of a denied URL"
    assert p.blocks("GET", "https://app.example/submit/1", "XHR") is None
    assert p.blocks("GET", "https://other.example/x", "Document") is None
    assert p.blocks("POST", "https://api.vendor.example/x", from_page=False) is None
    assert p.blocks("POST", "https://app.example/submit/1", from_page=False)
    q = Policy.from_dict({"deny_methods": ["delete"]})
    assert q.blocks("DELETE", "https://app.example/doc/1") == "DELETE is denied"
    assert q.blocks("PUT", "https://app.example/doc/1") is None


def test_policy_records_writes_and_matching_reads():
    p = Policy.from_dict({"record": ["/api/"]})
    assert p.records("PUT", "https://x/anything")
    assert p.records("GET", "https://x/api/doc")
    assert not p.records("GET", "https://x/style.css")
    assert not p.records("GET", "https://x/api/bundle.js", "Script")


def test_secrets_never_reach_the_log():
    p = Policy.from_dict({"redact": ["idp\\.example"]})
    assert p.body("https://idp.example/login", "user=a&x=1") == {"redacted": 10}
    for secret in ("username=a&password=hunter2", '{"user": "a", "Password": "x"}',
                   "j_password=x", "client_secret=abc", "SAMLResponse=PHN",
                   '<input value="eyJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJlM2UxODg.sig">', '{"accessToken" : "x"}', "pwd=1"):
        assert "redacted" in p.body("https://app.example/x", secret), secret
    assert p.body("https://app.example/x", '{"name":"x"}') == {"body": '{"name":"x"}'}
    # It errs on the safe side: a key that merely looks like a secret hides the body too.
    assert "redacted" in p.body("https://app.example/x", '{"passageNo": 3}')


def test_fault_hits_the_nth_matching_write_only():
    f = Fault.parse("transient_error:/api/doc:2")
    hits = [f.hit(m, u) for m, u in [("PUT", "https://x/api/doc"), ("GET", "https://x/api/doc"),
                                     ("PUT", "https://x/other"), ("PUT", "https://x/api/doc"),
                                     ("PUT", "https://x/api/doc")]]
    assert hits == [False, False, False, True, False]


def test_session_drop_holds_from_its_write_on():
    f = Fault.parse("session_drop:/api/")
    hits = [f.hit(m, u) for m, u in [("GET", "https://x/api/a"), ("PUT", "https://x/api/doc"),
                                     ("GET", "https://x/api/a"), ("GET", "https://x/style.css")]]
    assert hits == [False, True, True, False]


def test_rewrite_points_devtools_addresses_here():
    text = '{"webSocketDebuggerUrl": "ws://127.0.0.1:9335/devtools/browser/abc"}'
    assert "ws://127.0.0.1:9333/devtools" in rewrite(text, "127.0.0.1:9335", "127.0.0.1:9333")


# -- a real browser ----------------------------------------------------------------------------

CHROME = next((c for c in (
    os.environ.get("AJANTIK_CHROME"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
) if c and Path(c).exists()), None)

needs_chrome = pytest.mark.skipif(CHROME is None, reason="Chrome is not installed")

APP = b"""<!doctype html><title>app</title>
<button id="save">Save</button>
<button id="go" style="position:absolute;left:200px;top:200px;width:200px;height:40px">
  <span>Proceed to submission</span></button>
<script>
document.querySelector('#save').onclick = () =>
  fetch('/api/doc', {method: 'PUT', body: '{"name":"x"}'});
document.querySelector('#go').onclick = () => {
  document.title = 'submitted'; fetch('/submit/now', {method: 'POST', body: 'go'}); };
</script>"""


class Site:
    def __init__(self) -> None:
        self.hits: list[tuple[str, str]] = []
        site = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _reply(self, body: bytes, ctype: str = "text/html") -> None:
                site.hits.append((self.command, self.path))
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._reply(APP if self.path == "/app" else b"<title>other</title>")

            def do_PUT(self) -> None:
                self._reply(b'{"saved": true}', "application/json")

            do_POST = do_PUT

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


class Client:
    """The smallest DevTools client: enough to drive one tab the way Playwright does."""

    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=None).__enter__()
        self.ids = itertools.count(1)
        self.events: list[dict[str, Any]] = []

    def send(self, method: str, params: dict[str, Any] | None = None,
             session: str | None = None) -> dict[str, Any]:
        i = next(self.ids)
        msg: dict[str, Any] = {"id": i, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(self.ws.recv(timeout=20))
            if reply.get("id") == i:
                return reply
            self.events.append(reply)

    def event(self, method: str, timeout: float = 15) -> dict[str, Any]:
        end = time.time() + timeout
        while True:
            for e in self.events:
                if e.get("method") == method:
                    self.events.remove(e)
                    return e
            left = end - time.time()
            if left <= 0:
                raise TimeoutError(method)
            self.events.append(json.loads(self.ws.recv(timeout=left)))

    def open_tab(self, url: str) -> str:
        self.send("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True,
                                           "flatten": True})
        target = self.send("Target.createTarget", {"url": "about:blank"})["result"]["targetId"]
        while (e := self.event("Target.attachedToTarget"))["params"]["targetInfo"][
                "targetId"] != target:
            pass  # Chrome first reports the targets that were already there
        sid = e["params"]["sessionId"]
        self.send("Runtime.runIfWaitingForDebugger", session=sid)
        self.send("Page.enable", session=sid)
        self.send("Page.navigate", {"url": url}, sid)
        self.event("Page.loadEventFired")
        return sid

    def eval(self, sid: str, expr: str) -> Any:
        r = self.send("Runtime.evaluate", {"expression": expr, "awaitPromise": True,
                                           "returnByValue": True}, sid)
        return r["result"]["result"].get("value")

    def mouse_click(self, sid: str, x: float, y: float) -> None:
        for kind in ("mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y,
                                                   "button": "left", "clickCount": 1}, sid)


def _ws_url(base: str) -> str:
    with connect_http(base + "/json/version") as r:
        return json.loads(r.read())["webSocketDebuggerUrl"]


def connect_http(url: str):
    import urllib.request
    return urllib.request.urlopen(url, timeout=10)


@pytest.fixture
def browser(tmp_path: Path):
    profile = tmp_path / "profile"
    proc = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=0",
                             f"--user-data-dir={profile}", "--no-first-run",
                             "--no-default-browser-check", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file = profile / "DevToolsActivePort"
    for _ in range(100):
        if port_file.exists() and port_file.read_text().strip():
            break
        time.sleep(0.1)
    port = port_file.read_text().split()[0]
    yield proc, f"http://127.0.0.1:{port}"
    proc.kill()
    proc.wait()


def _start_guard(upstream: str, tmp_path: Path, policy: dict[str, Any], *extra: str):
    config = tmp_path / "guard.json"
    config.write_text(json.dumps(policy))
    log = tmp_path / "cdp.jsonl"
    proc = subprocess.Popen([sys.executable, "-m", "ajantik.cdp", "--upstream", upstream,
                             "--port", "0", "--config", str(config), "--log", str(log), *extra],
                            stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    assert line.startswith(READY_PREFIX), line
    return proc, line[len(READY_PREFIX):].strip(), log


POLICY = {"deny": ["/submit/", "/portal"], "deny_clicks": ["proceed to submission"],
          "record": ["/api/"]}


def _rows(log: Path, kind: str) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [r for r in map(json.loads, log.read_text().splitlines()) if r["kind"] == kind]


@needs_chrome
def test_guard_blocks_submission_and_records_saves(browser, tmp_path):
    _chrome, upstream = browser
    site = Site()
    guard, here, log = _start_guard(upstream, tmp_path, POLICY)
    try:
        skill = Client(_ws_url(here))
        sid = skill.open_tab(site.url + "/app")

        skill.eval(sid, "document.querySelector('#save').click(); "
                        "new Promise(r => setTimeout(r, 500))")
        skill.mouse_click(sid, 300, 220)                    # a real click on the denied button
        skill.eval(sid, "document.querySelector('#go span').click()")   # and a scripted one
        failed = skill.eval(sid, "fetch('/submit/now', {method: 'POST'})"
                                 ".then(() => 'sent', e => 'failed')")
        nav = skill.send("Page.navigate", {"url": site.url + "/portal"}, sid)
        time.sleep(0.5)

        assert ("PUT", "/api/doc") in site.hits
        assert not any(p.startswith(("/submit", "/portal")) for _, p in site.hits), site.hits
        assert skill.eval(sid, "document.title") == "app"
        assert failed == "failed"
        assert "blocked" in nav["error"]["message"]
        saves = [r for r in _rows(log, "http") if r["url"].endswith("/api/doc")]
        assert saves and saves[0]["status"] == 200
        assert saves[0]["request"]["body"] == '{"name":"x"}'
        assert saves[0]["response"]["body"] == '{"saved": true}'
        assert len(_rows(log, "click_blocked")) >= 2
        assert {r["reason"] for r in _rows(log, "blocked")} >= {
            "write to a denied URL", "page load of a denied URL"}
    finally:
        guard.terminate()
        guard.wait()


@needs_chrome
def test_guard_holds_for_a_driver_that_bypasses_the_proxy(browser, tmp_path):
    """A browser extension (Claude in Chrome) or a script on Chrome's own port never passes
    through the proxy; the guard sits in the browser, so it holds for them too."""
    _chrome, upstream = browser
    site = Site()
    guard, _here, log = _start_guard(upstream, tmp_path, POLICY)
    try:
        direct = Client(_ws_url(upstream))
        sid = direct.open_tab(site.url + "/app")
        time.sleep(0.3)
        direct.eval(sid, "document.querySelector('#go').click()")
        assert direct.eval(sid, "document.title") == "app"
        nav = direct.send("Page.navigate", {"url": site.url + "/portal"}, sid)
        time.sleep(0.5)
        assert not any(p.startswith(("/submit", "/portal")) for _, p in site.hits), site.hits
        assert nav["result"].get("errorText") == "net::ERR_BLOCKED_BY_CLIENT"
        assert _rows(log, "click_blocked")
    finally:
        guard.terminate()
        guard.wait()


@needs_chrome
def test_proxy_fails_closed_when_the_browser_goes(browser, tmp_path):
    chrome, upstream = browser
    guard, _here, log = _start_guard(upstream, tmp_path, POLICY)
    chrome.kill()
    assert guard.wait(timeout=15) == 1
    assert _rows(log, "guard_lost")


SAVE = ("fetch('/api/doc', {method: 'PUT', body: JSON.stringify({n: %d})})"
        ".then(r => r.status, e => 'failed')")


@needs_chrome
@pytest.mark.parametrize(("fault", "seen", "received"), [
    ("transient_error", [503, 200], [1]),        # the first save never arrives
    ("phantom_success", [204, 200], [1]),        # "saved", and nothing arrived
    ("phantom_failure", [500, 200], [0, 1]),     # it arrived, and the page was told it failed
    ("session_drop", [401, 401], []),            # logged out from the first save on
])
def test_faults_on_a_real_browser(browser, tmp_path, fault, seen, received):
    _chrome, upstream = browser
    site = Site()
    guard, here, log = _start_guard(upstream, tmp_path, POLICY, "--fault", f"{fault}:/api/doc")
    try:
        skill = Client(_ws_url(here))
        sid = skill.open_tab(site.url + "/app")
        got = [skill.eval(sid, SAVE % n) for n in range(2)]
        time.sleep(0.5)
        assert got == seen
        puts = [p for m, p in site.hits if m == "PUT"]
        assert len(puts) == len(received)
        faults = _rows(log, "fault")
        assert faults and faults[0]["fault"] == fault
        http = [r for r in _rows(log, "http") if r["url"].endswith("/api/doc")]
        assert [json.loads(r["request"]["body"])["n"] for r in http] == received
        assert all(r["status"] == 200 for r in http)  # the system's own answers, not ours
    finally:
        guard.terminate()
        guard.wait()
