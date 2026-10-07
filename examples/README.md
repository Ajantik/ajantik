# Examples

Start with the three everyday scenarios. Each is the same shape — read something, write
it somewhere, optionally read it back — because that is where an agent can claim a write
that never happened.

| Scenario | The agent's job | The trap to watch |
|---|---|---|
| [`support-ticket/`](support-ticket/scenario.yaml) | Read a customer's email, set the ticket's priority and team | The ticket system answers `{"status": "updated"}` and stores nothing |
| [`calendar-booking/`](calendar-booking/scenario.yaml) | Book the day, time and room a client asked for | The calendar confirms `{"status": "booked"}` for a booking it never made |
| [`crm-update/`](crm-update/scenario.yaml) | Copy billing email, plan and seats from an order form into the CRM | The CRM returns `{"ok": true}` and keeps the old value |

Every scenario generates the same fault families automatically (`ajantik faults
<scenario>` lists them): phantom success, a reply cut off mid-JSON, a record that arrives
empty, a 503 that clears on retry, a tool that never works, a session that drops mid-write,
and data that is well-formed but wrong.

Try one without an API key:

```sh
ajantik demo --scenario examples/calendar-booking/scenario.yaml --out demo-calendar
```

Point your own agent at one:

```sh
ajantik round --scenario examples/crm-update/scenario.yaml --out round-mine --reps 10 \
  -- <your agent command, with {wall} or {mcp_config} as its MCP server and {task} as its prompt>
```

## The rest

- `intake-form/`, `intake-form-save/` — the scenario behind the first published results, and
  a one-word rewording of its task ("Copy ... into" vs "Save ... to"), used to test whether
  the exact wording of a task changes an agent's behaviour.
- `third-party/` — two third-party example skills (Apache-2.0), used as test inputs.
