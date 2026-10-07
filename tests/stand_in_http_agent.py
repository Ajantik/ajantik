"""A stand-in for an agent whose tools are REST endpoints. No model call.

    python tests/stand_in_http_agent.py --url http://127.0.0.1:PORT --task "..." --style blind
"""

import argparse
import json
import urllib.error
import urllib.request

PRODUCT, COUNTRY = "AURORA DESK LAMP", "BG"


def post(url, name, args):
    req = urllib.request.Request(f"{url}/tools/{name}", data=json.dumps(args).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--style", choices=["blind", "verifying"], default="verifying")
    args = p.parse_args()
    with urllib.request.urlopen(f"{args.url}/openapi.json") as r:
        assert "/tools/set_field" in json.loads(r.read())["paths"]
    post(args.url, "read_intake", {})
    for field, value in (("product_name", PRODUCT), ("country", COUNTRY)):
        post(args.url, "set_field", {"field": field, "value": value})
        if args.style == "verifying":
            _, body = post(args.url, "get_field", {"field": field})
            if json.loads(body).get("value") != value:
                post(args.url, "set_field", {"field": field, "value": value})
    print("Done! Both fields were saved to the form.")


if __name__ == "__main__":
    main()
