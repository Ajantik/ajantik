"""The wall over HTTP, for agents whose tools are REST endpoints.

An agent that calls web APIs (OpenAPI actions, a requests-based tool, a no-code agent
builder) is tested by pointing its base URL here. Each scenario tool becomes

    POST {url}/tools/{name}     JSON body = the tool's arguments

and the description of all of them is served at `GET {url}/openapi.json` and
`GET {url}/tools`. A successful call answers 200 with the tool's reply as the body; a
failing one answers 503 with the failure text, which is how a real API would signal
it and what an HTTP client is built to notice.

Same fault semantics and session record as the MCP wall: underneath it is the same
`FaultyServer`. The final state is written when the process receives SIGTERM or
SIGINT, so the round runner stops it after the agent exits.

    python -m ajantik.http_wall --scenario examples/intake-form/scenario.yaml \
        --fault "phantom-success:set_field" --record /tmp/session.jsonl

The first line on stdout is `ajantik-wall listening on http://127.0.0.1:PORT`.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from ajantik.scenario import load_scenario
from ajantik.wall import FaultyServer, Transcript, pick_fault

READY_PREFIX = "ajantik-wall listening on "


def openapi(server: FaultyServer, base_url: str) -> dict[str, Any]:
    paths = {
        f"/tools/{t.name}": {"post": {
            "operationId": t.name,
            "summary": t.description,
            "requestBody": {"required": True, "content": {
                "application/json": {"schema": t.input_schema}}},
            "responses": {"200": {"description": "Tool reply"},
                          "503": {"description": "Tool failure"}},
        }}
        for t in server.scenario.tools
    }
    return {"openapi": "3.1.0", "info": {"title": "Tools", "version": "1"},
            "servers": [{"url": base_url}], "paths": paths}


def make_handler(wall: FaultyServer, lock: threading.Lock) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # stdout is the readiness channel
            pass

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            host = f"http://{self.headers.get('Host') or 'localhost'}"
            if self.path == "/openapi.json":
                self._send(200, json.dumps(openapi(wall, host)), "application/json")
            elif self.path == "/tools":
                self._send(200, json.dumps({"tools": wall.definitions()}), "application/json")
            else:
                self._send(404, json.dumps({"error": f"no such path: {self.path}"}),
                           "application/json")

        def do_POST(self) -> None:
            if not self.path.startswith("/tools/"):
                self._send(404, json.dumps({"error": f"no such path: {self.path}"}),
                           "application/json")
                return
            name = self.path[len("/tools/"):]
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                arguments = json.loads(raw or b"{}")
            except ValueError:
                self._send(400, json.dumps({"error": "body is not JSON"}), "application/json")
                return
            if not isinstance(arguments, dict):
                self._send(400, json.dumps({"error": "body must be a JSON object"}),
                           "application/json")
                return
            with lock:  # one call at a time: the recorded order is the order of effect
                text, is_error = wall.call_tool(name, arguments)
            try:
                json.loads(text)
                ctype = "application/json"
            except ValueError:
                ctype = "text/plain"
            self._send(503 if is_error else 200, text, ctype)

    return Handler


def serve(wall: FaultyServer, host: str = "127.0.0.1", port: int = 0) -> int:
    lock = threading.Lock()
    httpd = HTTPServer((host, port), make_handler(wall, lock))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"{READY_PREFIX}http://{host}:{httpd.server_address[1]}", flush=True)
    stop.wait()
    httpd.shutdown()
    with lock:
        wall.finish()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ajantik.http_wall", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--fault", default="clean", help="fault id to apply (default: none)")
    p.add_argument("--record", type=Path, help="JSONL session record to write")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = p.parse_args(argv)
    scen = load_scenario(args.scenario)
    wall = FaultyServer(scen, pick_fault(scen, args.fault),
                        Transcript(args.record) if args.record else Transcript())
    return serve(wall, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
