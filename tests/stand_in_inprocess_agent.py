"""A stand-in for an agent built in code, using the in-process wall. No model call.

Run by `ajantik round --in-process`, which sets the variables Wall.from_env() reads.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ajantik.adapter import Wall

PRODUCT, COUNTRY = "AURORA DESK LAMP", "BG"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--style", choices=["blind", "verifying"], default="verifying")
    args = p.parse_args()
    with Wall.from_env() as wall:
        assert {t["function"]["name"] for t in wall.tools("openai")} >= {"set_field", "get_field"}
        wall.call("read_intake", {})
        for field, value in (("product_name", PRODUCT), ("country", COUNTRY)):
            wall.call("set_field", {"field": field, "value": value})
            if args.style == "verifying":
                text, _ = wall.call("get_field", {"field": field})
                if json.loads(text).get("value") != value:
                    wall.call("set_field", {"field": field, "value": value})
    print("Done! Both fields were saved to the form.")


if __name__ == "__main__":
    main()
