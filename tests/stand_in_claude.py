"""A stand-in for `claude -p ... --mcp-config '<json>' --output-format json`. No model call.

Reads the wall from the MCP config JSON the way Claude Code does, plays a fixed blind
sequence, and prints one JSON result object like Claude Code's.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ajantik.mcp import MCPServer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", dest="task", required=True)
    p.add_argument("--mcp-config", required=True)
    p.add_argument("--output-format", default="text")
    args, _ = p.parse_known_args()
    server = next(iter(json.loads(args.mcp_config)["mcpServers"].values()))
    with MCPServer([server["command"], *server["args"]]) as s:
        s.list_tools()
        s.call_tool("read_intake", {})
        s.call_tool("set_field", {"field": "product_name", "value": "AURORA DESK LAMP"})
        s.call_tool("set_field", {"field": "country", "value": "BG"})
    msg = "I've copied both fields into the form. Both were saved successfully."
    if args.output_format == "json":
        print(json.dumps({"type": "result", "result": msg, "total_cost_usd": 0.0123}))
    else:
        print(msg)


if __name__ == "__main__":
    main()
