"""F3: the trust boundary, re-expressed as framework hooks.

The Part A design is unchanged and unchanged *on purpose*: the capability set is
built from the operator's arguments before `agent.run()` is called, carried in
`deps`, and never re-read from anything the model says or a tool returns. The
framework moves where the check runs, not what it decides.

Where you can intervene (all confirmed against pydantic_ai 2.35):

    prepare_tools          before the tool schemas are sent -- withhold a tool
                           from the model entirely
    before_tool_validate   raw arguments, before parsing
    before_tool_execute    validated arguments, before the body runs  <- used here
    wrap_tool_execute      around the body, both sides   <- used by durability.py
    after_tool_execute     the return value, before it becomes a ToolReturnPart

`before_tool_execute` is the right seam for a capability check: arguments are
already validated, and raising here means the body never runs, so a refused
`send_email` cannot reach the ledger.

Where you *cannot* intervene, and it costs something:

  * There is no hook between "the model emitted a tool call" and "the framework
    decided to run it" that can cancel the call while leaving the transcript
    well-formed. Refusal has to be an exception raised from inside the tool
    path, which the framework then renders as a retry prompt.

  * Raising `ModelRetry` spends the tool's retry budget. A policy refusal is not
    a transient failure and should not consume the allowance reserved for
    genuinely malformed arguments, so refusals here return a legible *string*
    instead, and the tool budget is left for S2.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from pydantic_ai.tools import RunContext

from agent.paths import PathDenied, safe_path
from agent.policy import PolicyDenied

from .deps import FwDeps


class Refusal(Exception):
    """A capability check failed. Carries text written for the model to read."""


def enforce(ctx: RunContext[FwDeps], tool: str, args: dict[str, Any]) -> None:
    """Raise `Refusal` if the frozen policy does not permit this call.

    Called from `before_tool_execute`, so nothing downstream of it runs.
    """
    policy = ctx.deps.policy
    try:
        if tool == "send_email":
            policy.check_email(str(args.get("to", "")))
        elif tool == "write_file":
            policy.check_write()
            safe_path(policy.workspace, str(args.get("path", "")))
        elif tool == "read_file":
            safe_path(policy.workspace, str(args.get("path", "")))
        elif tool == "run_python":
            policy.check_python()
        elif tool == "http_get":
            parsed = urllib.parse.urlparse(str(args.get("url", "")))
            if parsed.scheme not in ("http", "https"):
                raise PolicyDenied(
                    f"http_get refused: scheme {parsed.scheme!r} is not allowed. "
                    "Only http and https URLs may be fetched."
                )
            policy.check_http_host(parsed.hostname)
    except (PolicyDenied, PathDenied) as exc:
        ctx.deps.denials += 1
        ctx.deps.record(
            ctx.run_step,
            "policy_denied",
            {"tool": tool, "args": args, "reason": str(exc), "call_id": ctx.tool_call_id},
        )
        raise Refusal(str(exc)) from exc
