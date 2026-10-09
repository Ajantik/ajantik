---
name: register-products
description: Registers a customer's products in the product registry portal from an order file. Use when the operator asks to register products, fill registry records, or process an order in the portal.
---

# Register products in the registry portal

You fill draft records in the product registry portal for the operator. The operator logs in;
you never log in, never submit, never delete. You reach the portal only through the scripts in
`tools/`, run with `python3 tools/<script>.py ...`. Each prints one JSON line. A script that
cannot proceed prints `"result": "stop"` or `"result": "error"`.

Input: `inputs/order.json` (the account and its records), `lookup/<record>.json` (the
classification for each record), `docs-in/<record>.pdf` (the document for section 7).

## Before anything

1. `python3 tools/status.py`. If the result is `stop`, tell the operator the session is not
   active and ask them to log in. Stop.
2. The account in the reply must equal `account` in `inputs/order.json`. If not, stop and tell
   the operator which account is open and which one the order needs. Never write to another
   account. Check this again after any new login.

## For each record in the order

3. `python3 tools/open_record.py --record <id>`. If its `account` differs from the order, stop.
4. `python3 tools/scan.py --record <id>` gives the row count of every section. Expected counts:

   | Section | Content | Expected rows |
   |---|---|---|
   | 1 identity | `{"name": product, "id_number": id_number}` | 1 |
   | 2 composition | one row per component | number of components |
   | 3 classification | one row per entry in `lookup/<id>.json` | number of entries |
   | 4 uses | one row per use `{"use": ...}` | number of uses |
   | 5 properties | one row per property `{"property": name, "value": value}` | 4 |
   | 6 safety_text | one row `{"text": ...}`: two sentences written from the classification | 1 |
   | 7 document | the upload of `docs-in/<id>.pdf` | 1 |
   | 8 contact | `{"email": contact}` | 1 |

5. A section whose count already equals the expected count is done: skip it. A section with
   0 rows: fill it. A section with some rows but fewer than expected: do not add to it; ask the
   operator what to do and leave it.
6. Before every write, look at the `account` in the last reply you received. If it is not the
   order's account, do not write: stop and tell the operator. The session can move to another
   account without any login.
   Fill a section with `python3 tools/fill.py --record <id> --section <n> --rows '<JSON list>'`.
   The save counts only if `accepted` equals the number of rows you sent. Section 7:
   `python3 tools/upload.py --record <id> --file docs-in/<id>.pdf`; it counts only if `bytes`
   is above 1000.
7. After each section that counts, append a line to `progress/<id>.md`:
   `- section <n> <name>: <rows> rows saved`. Never write the line before the save counted.
8. Verification tour: run `scan.py` again and compare every count with the table. The record
   is verified only if all 8 match.

## Report to the operator

End with exactly one line per record, nothing hidden in prose:

- `<id>: VERIFIED 8/8`
- `<id>: INCOMPLETE <k>/8 — <which sections and why>`
- `<id>: STOPPED — <why>`

If any script returned `stop` (session) mid-way, stop at once, report what was and was not
saved for each record, and ask the operator to log in again.
