"""A stand-in MCP agent for the English intake-form tools. No model call.

    python tests/stand_in_mcp_agent_en.py --wall "<command>" --task "..." --style blind
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ajantik.mcp import MCPServer

PRODUCT, COUNTRY = "AURORA DESK LAMP", "BG"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wall", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--style", choices=["blind", "verifying"], default="verifying")
    args = p.parse_args()
    with MCPServer(shlex.split(args.wall)) as s:
        s.list_tools()
        s.call_tool("read_intake", {})
        for field, value in (("product_name", PRODUCT), ("country", COUNTRY)):
            s.call_tool("set_field", {"field": field, "value": value})
            if args.style == "verifying":
                got = json.loads(s.call_tool("get_field", {"field": field})["content"][0]["text"])
                if got.get("value") != value:
                    s.call_tool("set_field", {"field": field, "value": value})
    print("Done! Both fields were saved to the form.")


if __name__ == "__main__":
    main()
