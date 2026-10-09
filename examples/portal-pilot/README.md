# Portal pilot: a long, multi-stage, operator-supervised skill (synthetic)

A fictional product registry portal and a Claude Code skill that fills it, in the shape of
real portal-filling skills: the agent reaches the portal only through scripts it runs in its
shell (one JSON line each), saves row by row with an acceptance count, uploads a document,
runs a verification tour, keeps a progress file and ends with a structured report. The
operator logs in; sessions end when idle; every session belongs to one account.

Everything here is invented. It exists so that Ajantik can be developed against a big skill
whose tool layer is not MCP.

```sh
cd examples/portal-pilot
export PORTAL_HOME=$PWD/portal-data
python3 tools/login.py --account ACME          # the operator's step
claude                                         # then: "register the products in inputs/order.json"
python3 tools/scan.py --record R-102           # what the portal really holds
```

`rm -rf portal-data progress/*.md` resets everything.
