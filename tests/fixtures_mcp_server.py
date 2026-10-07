#!/usr/bin/env python3
"""A tiny, stateful MCP stdio server used only by the tests.

Not a general MCP implementation. It has one write tool and one read tool over an
in-memory store, so a test can prove whether a call really reached the server: if a
phantom fault forwarded the write, `read_all` would show it.
"""

import json
import sys

STORE: dict = {}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method, rpc_id = msg.get("method"), msg.get("id")

        if method == "initialize":
            result = {"protocolVersion": "2025-06-18",
                      "serverInfo": {"name": "fake-store", "version": "0"},
                      "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": [
                {"name": "store_value", "description": "store a value",
                 "inputSchema": {"type": "object",
                                 "properties": {"field": {"type": "string"},
                                                "value": {"type": "string"}}}},
                {"name": "read_all", "description": "read the whole store",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}
        elif method == "tools/call":
            params = msg.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            if name == "store_value":
                STORE[str(args.get("field"))] = args.get("value")
                result = {"content": [{"type": "text", "text": "Stored (real)."}]}
            elif name == "read_all":
                result = {"content": [{"type": "text",
                                       "text": json.dumps(STORE, ensure_ascii=False)}]}
            else:
                _send({"jsonrpc": "2.0", "id": rpc_id,
                       "error": {"code": -32602, "message": "unknown tool"}})
                continue
        elif rpc_id is None:
            continue  # a notification needs no reply
        else:
            _send({"jsonrpc": "2.0", "id": rpc_id,
                   "error": {"code": -32601, "message": f"no method {method}"}})
            continue
        _send({"jsonrpc": "2.0", "id": rpc_id, "result": result})


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
