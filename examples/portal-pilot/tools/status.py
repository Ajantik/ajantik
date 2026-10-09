"""Is a session open, and for which account? Read only."""

import portal

state = portal.load()
account, problem = portal.session(state)
portal.save(state)
if problem:
    portal.out({"result": "stop", "stop": problem,
                "message": "The session is not active. The operator must log in again."})
else:
    portal.out({"result": "ok", "account": account})
