"""CDP guard: the boundary for skills that drive a real browser over the DevTools protocol.

A skill that runs a portal through Playwright (`chromium.connectOverCDP(CDP_URL)`) reaches the
world through one Chrome, the one with the user's login. `ajantik cdp` stands in front of it:

    skill (Playwright) --CDP_URL--> ajantik cdp --> Chrome (--remote-debugging-port=UPSTREAM)
                                        |
                                        +-- its own guard session on every tab, frame, worker

The guard session is Ajantik's own DevTools client, attached to every page, frame and worker
of the browser, old and new. Through it:

  block   a write (any method but GET/HEAD/OPTIONS) whose URL matches a `deny` pattern, or
          does not match any `allow_writes` pattern (when that list is given), and a page load
          whose URL matches `deny`, fail with net::ERR_BLOCKED_BY_CLIENT. A click, press or
          form submit on a control whose label matches a `deny_clicks` pattern is swallowed
          before the page sees it. Every block is logged.
  record  every write, and every XHR/fetch read whose URL matches a `record` pattern: method,
          URL, request body, status and response body (up to `body_limit` bytes each), as JSONL.

The skill's own connection passes through here. No command it sends reaches a tab before the
guard is installed on that tab, and when the guard's connection to Chrome is lost, the skill's
connections are closed and the process exits (fail closed). Start Chrome on a port only
Ajantik is told about and give the skill this address, so there is no way around it.

    python -m ajantik.cdp --upstream http://127.0.0.1:9335 --port 9333 \\
        --config guard.json --log /tmp/cdp.jsonl

Config (JSON, every key optional):

    {"deny": ["regex", ...], "allow_writes": ["regex", ...], "deny_clicks": ["regex", ...],
     "record": ["regex", ...], "body_limit": 65536}

The first line on stdout is `ajantik-cdp listening on http://HOST:PORT`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, ClassVar

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

READY_PREFIX = "ajantik-cdp listening on "
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
GUARDED_TYPES = {"page", "iframe", "background_page", "webview"}  # guard in place before the skill
WORKER_TYPES = {"worker", "shared_worker", "service_worker"}
WORLD = "ajantik_guard"                      # the isolated world the click guard lives in
BINDING = "__ajantikBlocked"
READY_TIMEOUT = 10.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


@dataclass
class Policy:
    deny: list[re.Pattern[str]] = field(default_factory=list)
    allow_writes: list[re.Pattern[str]] | None = None
    deny_clicks: list[str] = field(default_factory=list)   # JavaScript regex sources
    record: list[re.Pattern[str]] = field(default_factory=list)
    body_limit: int = 65536

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Policy:
        allow = d.get("allow_writes")
        return cls(deny=_compile(d.get("deny", [])),
                   allow_writes=_compile(allow) if allow is not None else None,
                   deny_clicks=list(d.get("deny_clicks", [])),
                   record=_compile(d.get("record", [])),
                   body_limit=int(d.get("body_limit", 65536)))

    def blocks(self, method: str, url: str, resource_type: str | None = None,
               from_page: bool = True) -> str | None:
        """Why this request must not leave the browser, or None. `allow_writes` holds for what
        pages send; a worker or an extension's own background (Claude in Chrome talking to its
        server) is held to `deny` only."""
        denied = any(p.search(url) for p in self.deny)
        if method.upper() not in SAFE_METHODS:
            if denied:
                return "write to a denied URL"
            if (from_page and self.allow_writes is not None
                    and not any(p.search(url) for p in self.allow_writes)):
                return "write outside allow_writes"
            return None
        if denied and resource_type == "Document":
            return "page load of a denied URL"
        return None

    def records(self, method: str, url: str, resource_type: str | None = None) -> bool:
        """Every write; a read only when a script asked for it (not the page's own assets)."""
        if method.upper() not in SAFE_METHODS:
            return True
        return resource_type in (None, "XHR", "Fetch") and any(p.search(url) for p in self.record)

    def summary(self) -> str:
        parts = [f"{len(self.deny)} deny", f"{len(self.deny_clicks)} deny_clicks",
                 f"{len(self.record)} record"]
        if self.allow_writes is not None:
            parts.insert(1, f"{len(self.allow_writes)} allow_writes")
        return ", ".join(parts)


def click_guard(patterns: list[str]) -> str:
    """The script that swallows clicks on denied controls. It runs in an isolated world, so the
    page cannot see or remove it; DOM events are shared, so stopping one here stops it for the
    page too. Capture listeners on window run before any listener the page puts on an element."""
    return """(() => {
  if (globalThis.__ajantikGuard) return; globalThis.__ajantikGuard = 1;
  const RX = PATTERNS.map(s => new RegExp(s, 'i'));
  const CONTROL = 'button, a, [role="button"], [role="link"], [role="menuitem"], [role="tab"],'
    + ' input[type="submit"], input[type="button"], input[type="image"], input[type="reset"]';
  const label = el => [el.innerText, el.value, el.getAttribute('aria-label'),
    el.getAttribute('title'), el.getAttribute('data-e2e')]
    .filter(x => typeof x === 'string' && x).join(' ').replace(/\\s+/g, ' ').trim();
  const hit = node => {
    let el = node && node.closest ? node.closest(CONTROL) : null;
    while (el) {
      const t = label(el);
      if (RX.some(r => r.test(t))) return t;
      el = el.parentElement ? el.parentElement.closest(CONTROL) : null;
    }
    return null;
  };
  const KEYS = new Set(['Enter', ' ', 'Spacebar']);
  const swallow = e => {
    if (e.type.startsWith('key') && !KEYS.has(e.key)) return;
    const node = e.type === 'submit' ? (e.submitter || e.target)
      : (e.composedPath ? e.composedPath()[0] : e.target);
    const t = hit(node);
    if (!t) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if ((e.type === 'click' || e.type === 'submit') && globalThis.BINDING)
      globalThis.BINDING(JSON.stringify({event: e.type, label: t.slice(0, 200), url: location.href}));
  };
  for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'dblclick',
                      'auxclick', 'keydown', 'keypress', 'keyup', 'submit', 'touchstart',
                      'touchend'])
    window.addEventListener(type, swallow, true);
})();""".replace("BINDING", BINDING).replace("PATTERNS", json.dumps(patterns))


class Log:
    def __init__(self, path: Path | None):
        self.path = path
        self.counts: Counter[str] = Counter()

    def write(self, kind: str, **row: Any) -> None:
        self.counts[kind] += 1
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": _now(), "kind": kind, **row}, ensure_ascii=False) + "\n")


def _body(text: str | bytes | None, limit: int) -> dict[str, Any]:
    if text is None:
        return {}
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytes": len(text), "binary": True}
    out: dict[str, Any] = {"body": text[:limit]}
    if len(text) > limit:
        out["truncated"] = len(text)
    return out


class CDPError(Exception):
    pass


class Guard:
    """Ajantik's own DevTools session on every target of one browser."""

    def __init__(self, ws_url: str, policy: Policy, log: Log):
        self.ws_url = ws_url
        self.policy = policy
        self.log = log
        self.ws: ClientConnection | None = None
        self.closed = asyncio.Event()
        self._id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._targets: dict[str, dict[str, Any]] = {}     # guard session id -> targetInfo
        self._open: dict[tuple[str, str], dict[str, Any]] = {}  # (session, network id) -> row
        self._tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        self.ws = await connect(self.ws_url, max_size=None, ping_interval=None, compression=None,
                                proxy=None)
        self._spawn(self._reader())
        await self.send("Target.setAutoAttach", {"autoAttach": True,
                                                  "waitForDebuggerOnStart": True,
                                                  "flatten": True})

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send(self, method: str, params: dict[str, Any] | None = None,
                   session: str | None = None) -> dict[str, Any]:
        assert self.ws is not None
        self._id += 1
        msg: dict[str, Any] = {"id": self._id, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[self._id] = fut
        await self.ws.send(json.dumps(msg))
        return await fut

    async def _reader(self) -> None:
        assert self.ws is not None
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if "error" in msg:
                            fut.set_exception(CDPError(msg["error"].get("message", "error")))
                        else:
                            fut.set_result(msg.get("result", {}))
                elif (handler := self._handlers.get(msg.get("method", ""))) is not None:
                    self._spawn(handler(self, msg.get("params", {}), msg.get("sessionId")))
        except ConnectionClosed:
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(CDPError("guard connection closed"))
            self.closed.set()

    # -- targets -----------------------------------------------------------------------------

    def ready_event(self, target_id: str) -> asyncio.Event:
        return self._ready.setdefault(target_id, asyncio.Event())

    async def wait_ready(self, target_id: str, timeout: float = READY_TIMEOUT) -> bool:
        try:
            await asyncio.wait_for(self.ready_event(target_id).wait(), timeout)
            return True
        except TimeoutError:
            return False

    async def _on_attached(self, p: dict[str, Any], _parent: str | None) -> None:
        sid, info = p["sessionId"], p["targetInfo"]
        self._targets[sid] = info
        kind = info.get("type")
        try:
            if kind in GUARDED_TYPES:
                await self._install_page(sid)
            elif kind in WORKER_TYPES:
                await self._install_worker(sid)
            if p.get("waitingForDebugger"):
                await self.send("Runtime.runIfWaitingForDebugger", session=sid)
        except CDPError as e:
            if sid in self._targets:  # still there: the guard could not take its place
                self.log.write("guard_error", target=info.get("url"), type=kind, error=str(e))
                return
        if kind in GUARDED_TYPES:
            self.ready_event(info["targetId"]).set()
            self.log.write("guarded", type=kind, url=info.get("url"), target=info["targetId"],
                           session=sid, waiting=bool(p.get("waitingForDebugger")))
            if self.policy.deny_clicks:
                # Documents already there (or committing while we installed) get the click
                # guard now; a paused tab cannot run it until resumed, so this comes after.
                tree = await self._quiet("Page.getFrameTree", {}, sid) or {}
                for frame_id in _frame_ids(tree.get("frameTree", {})):
                    await self._inject(sid, frame_id)

    async def _install_page(self, sid: str) -> None:
        await self.send("Fetch.enable", {"patterns": [{"urlPattern": "*",
                                                       "requestStage": "Request"}]}, sid)
        await self.send("Network.enable", {}, sid)
        if self.policy.deny_clicks:
            await self.send("Page.enable", {}, sid)  # new-document scripts need it
            await self.send("Runtime.enable", {}, sid)
            await self.send("Runtime.addBinding", {"name": BINDING,
                                                   "executionContextName": WORLD}, sid)
            await self.send("Page.addScriptToEvaluateOnNewDocument",
                            {"source": click_guard(self.policy.deny_clicks),
                             "worldName": WORLD}, sid)
        await self.send("Target.setAutoAttach", {"autoAttach": True,
                                                  "waitForDebuggerOnStart": True,
                                                  "flatten": True}, sid)

    async def _quiet(self, method: str, params: dict[str, Any],
                     sid: str) -> dict[str, Any] | None:
        try:
            return await self.send(method, params, sid)
        except CDPError:
            return None  # the page or frame went away

    async def _inject(self, sid: str, frame_id: str) -> None:
        """The click guard into one frame's current document (a no-op where it already is)."""
        world = await self._quiet("Page.createIsolatedWorld",
                                  {"frameId": frame_id, "worldName": WORLD}, sid)
        if world:
            await self._quiet("Runtime.evaluate",
                              {"expression": click_guard(self.policy.deny_clicks),
                               "contextId": world["executionContextId"]}, sid)

    async def _on_navigated(self, p: dict[str, Any], sid: str | None) -> None:
        # A navigation that started before the new-document script was registered commits
        # without it; this closes that gap.
        if sid and self.policy.deny_clicks and "frame" in p:
            await self._inject(sid, p["frame"]["id"])

    async def _install_worker(self, sid: str) -> None:
        for method, params in (("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}),
                               ("Network.enable", {})):
            try:
                await self.send(method, params, sid)
            except CDPError:
                pass  # not every worker kind speaks every domain

    async def _on_detached(self, p: dict[str, Any], _parent: str | None) -> None:
        info = self._targets.pop(p.get("sessionId", ""), None)
        if info:
            self._ready.pop(info.get("targetId", ""), None)

    # -- requests ----------------------------------------------------------------------------

    async def _on_paused(self, p: dict[str, Any], sid: str | None) -> None:
        req = p["request"]
        method, url, rtype = req["method"], req["url"], p.get("resourceType")
        info = self._targets.get(sid or "", {})
        from_page = info.get("type") in ("page", "iframe", "webview") and not str(
            info.get("url", "")).startswith("chrome-extension://")
        reason = self.policy.blocks(method, url, rtype, from_page)
        try:
            if reason:
                await self.send("Fetch.failRequest", {"requestId": p["requestId"],
                                                      "errorReason": "BlockedByClient"}, sid)
                self.log.write("blocked", reason=reason, method=method, url=url, type=rtype,
                               **_body(_post_data(req), self.policy.body_limit))
                return
            if self.policy.records(method, url, rtype) and sid and p.get("networkId"):
                self._open[(sid, p["networkId"])] = {
                    "method": method, "url": url, "type": rtype,
                    "request": _body(_post_data(req), self.policy.body_limit)}
            await self.send("Fetch.continueRequest", {"requestId": p["requestId"]}, sid)
        except CDPError:
            pass  # the page or the request went away

    async def _on_response(self, p: dict[str, Any], sid: str | None) -> None:
        row = self._open.get((sid or "", p["requestId"]))
        if row is not None:
            row["status"] = p["response"].get("status")

    async def _on_finished(self, p: dict[str, Any], sid: str | None) -> None:
        row = self._open.pop((sid or "", p["requestId"]), None)
        if row is None:
            return
        try:
            got = await self.send("Network.getResponseBody", {"requestId": p["requestId"]}, sid)
            body = base64.b64decode(got["body"]) if got.get("base64Encoded") else got["body"]
            row["response"] = _body(body, self.policy.body_limit)
        except CDPError:
            row["response"] = {}
        self.log.write("http", **row)

    async def _on_failed(self, p: dict[str, Any], sid: str | None) -> None:
        row = self._open.pop((sid or "", p["requestId"]), None)
        if row is not None:
            self.log.write("http", **row, error=p.get("errorText"))

    async def _on_binding(self, p: dict[str, Any], sid: str | None) -> None:
        if p.get("name") != BINDING:
            return
        try:
            data = json.loads(p.get("payload") or "{}")
        except ValueError:
            data = {"payload": p.get("payload")}
        self.log.write("click_blocked", **data)

    _handlers: ClassVar[dict[str, Any]] = {
        "Target.attachedToTarget": _on_attached,
        "Target.detachedFromTarget": _on_detached,
        "Fetch.requestPaused": _on_paused,
        "Network.responseReceived": _on_response,
        "Network.loadingFinished": _on_finished,
        "Network.loadingFailed": _on_failed,
        "Runtime.bindingCalled": _on_binding,
        "Page.frameNavigated": _on_navigated,
    }


def _post_data(req: dict[str, Any]) -> str | bytes | None:
    if "postData" in req:
        return req["postData"]
    entries = req.get("postDataEntries")
    if entries:
        return b"".join(base64.b64decode(e.get("bytes", "")) for e in entries)
    return None


def _frame_ids(tree: dict[str, Any]) -> list[str]:
    ids = [tree["frame"]["id"]] if "frame" in tree else []
    for child in tree.get("childFrames", []):
        ids += _frame_ids(child)
    return ids


# -- the skill's side ----------------------------------------------------------------------

LOGGED_COMMANDS = {"Page.navigate", "Page.reload", "Target.createTarget", "Target.closeTarget"}


def rewrite(text: str, upstream: str, here: str) -> str:
    """Point the DevTools addresses Chrome hands out at us instead of at Chrome."""
    return text.replace(upstream, here)


class Proxy:
    def __init__(self, upstream: str, guard: Guard, policy: Policy, log: Log):
        self.upstream = upstream.rstrip("/")                  # http://127.0.0.1:9335
        self.upstream_hostport = self.upstream.split("://", 1)[1]
        self.guard = guard
        self.policy = policy
        self.log = log
        self.here = ""                                        # host:port, set by run()
        self.clients: set[ServerConnection] = set()

    def _http(self, path: str) -> str:
        with urllib.request.urlopen(self.upstream + path, timeout=10) as r:
            return r.read().decode("utf-8")

    async def process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        path = request.path
        if path.startswith("/devtools/"):
            if self.guard.closed.is_set():
                return conn.respond(HTTPStatus.SERVICE_UNAVAILABLE, "ajantik: guard is down\n")
            return None  # a WebSocket upgrade: handled by `handler`
        if path.split("?")[0].rstrip("/") in ("/json/version", "/json", "/json/list"):
            try:
                text = await asyncio.to_thread(self._http, path)
            except OSError as e:
                return conn.respond(HTTPStatus.BAD_GATEWAY, f"ajantik: Chrome unreachable: {e}\n")
            body = rewrite(text, self.upstream_hostport, self.here).encode("utf-8")
            return Response(200, "OK", Headers([("Content-Type", "application/json"),
                                                 ("Content-Length", str(len(body)))]), body)
        return conn.respond(HTTPStatus.NOT_FOUND, "ajantik: not a DevTools path\n")

    async def handler(self, client: ServerConnection) -> None:
        url = "ws://" + self.upstream_hostport + client.request.path
        try:
            chrome = await connect(url, max_size=None, ping_interval=None, compression=None,
                                   proxy=None)
        except (OSError, ConnectionClosed) as e:
            await client.close(1011, f"ajantik: Chrome unreachable: {e}"[:120])
            return
        self.clients.add(client)
        sessions: dict[str, tuple[str, str]] = {}   # skill's session id -> (targetId, type)
        try:
            await _first_done(self._to_chrome(client, chrome, sessions),
                              self._to_skill(chrome, client, sessions))
        finally:
            self.clients.discard(client)
            await chrome.close()
            await client.close()

    async def _to_skill(self, chrome: ClientConnection, client: ServerConnection,
                        sessions: dict[str, tuple[str, str]]) -> None:
        async for raw in chrome:
            if '"Target.attachedToTarget"' in raw:
                msg = json.loads(raw)
                p = msg.get("params", {})
                if msg.get("method") == "Target.attachedToTarget":
                    info = p.get("targetInfo", {})
                    sessions[p["sessionId"]] = (info.get("targetId", ""), info.get("type", ""))
            await client.send(raw)

    async def _to_chrome(self, client: ServerConnection, chrome: ClientConnection,
                         sessions: dict[str, tuple[str, str]]) -> None:
        async for raw in client:
            msg = json.loads(raw)
            method, sid = msg.get("method", ""), msg.get("sessionId")
            target = sessions.get(sid or "")
            if target and target[1] in GUARDED_TYPES and not await self.guard.wait_ready(target[0]):
                await self._refuse(client, msg, "the guard is not in place on this tab")
                continue
            url = (msg.get("params") or {}).get("url")
            if method in ("Page.navigate", "Target.createTarget") and url:
                reason = self.policy.blocks("GET", url, "Document")
                if reason:
                    self.log.write("blocked", reason=reason, method="NAVIGATE", url=url,
                                   via=method)
                    await self._refuse(client, msg, f"blocked: {reason}")
                    continue
            if method in LOGGED_COMMANDS:
                self.log.write("cdp", method=method, url=url, target=target[0] if target else None)
            await chrome.send(raw)

    async def _refuse(self, client: ServerConnection, msg: dict[str, Any], why: str) -> None:
        reply: dict[str, Any] = {"id": msg.get("id"),
                                 "error": {"code": -32000, "message": f"ajantik: {why}"}}
        if msg.get("sessionId"):
            reply["sessionId"] = msg["sessionId"]
        await client.send(json.dumps(reply))


async def _first_done(*coros: Any) -> None:
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def browser_ws(upstream: str) -> str:
    with urllib.request.urlopen(upstream.rstrip("/") + "/json/version", timeout=10) as r:
        return json.loads(r.read())["webSocketDebuggerUrl"]


async def run(upstream: str, host: str, port: int, policy: Policy, log: Log,
              ready: Any = None) -> int:
    """Guard the browser at `upstream` and serve the skill at host:port until the guard's
    connection ends (exit 1) or the task is cancelled (exit 0)."""
    guard = Guard(await asyncio.to_thread(browser_ws, upstream), policy, log)
    await guard.start()
    proxy = Proxy(upstream, guard, policy, log)
    async with serve(proxy.handler, host, port, process_request=proxy.process_request,
                     max_size=None, ping_interval=None, compression=None) as server:
        bound = server.sockets[0].getsockname()[1]
        proxy.here = f"{host}:{bound}"
        print(f"{READY_PREFIX}http://{proxy.here}", flush=True)
        if ready:
            ready(proxy.here)
        try:
            await guard.closed.wait()
        except asyncio.CancelledError:
            return 0
        log.write("guard_lost")
        for c in list(proxy.clients):
            await c.close(1011, "ajantik: guard lost, closing (fail closed)")
        return 1


def load_policy(config: Path | None, **extra: list[str]) -> Policy:
    data: dict[str, Any] = json.loads(config.read_text(encoding="utf-8")) if config else {}
    for key, values in extra.items():
        if values:
            data[key] = [*(data.get(key) or []), *values]
    return Policy.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ajantik.cdp", description=__doc__.split("\n")[0])
    ap.add_argument("--upstream", default="http://127.0.0.1:9335")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--deny", action="append", default=[])
    ap.add_argument("--allow-write", action="append", default=[])
    ap.add_argument("--deny-click", action="append", default=[])
    ap.add_argument("--record", action="append", default=[])
    a = ap.parse_args(argv)
    policy = load_policy(a.config, deny=a.deny, allow_writes=a.allow_write,
                         deny_clicks=a.deny_click, record=a.record)
    try:
        return asyncio.run(run(a.upstream, a.host, a.port, policy, Log(a.log)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
