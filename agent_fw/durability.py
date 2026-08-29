"""F2: exactly-once side effects around a loop we do not own.

Part A could put persistence exactly where it wanted, because it wrote the loop.
Here the framework decides when the model is called and when a tool runs, so the
guarantee has to be reassembled from two seams it does offer.

**Seam 1 -- `wrap_tool_execute`.** Wraps every tool body. For an irreversible
tool this checks the ledger first and, if the key is already committed, returns
the recorded result *without calling the handler*. Otherwise it runs the body and
commits the effect and its ledger row in one transaction, exactly as Part A did.
`agent.storage.commit_email` is reused verbatim: the SQL guarantee is unchanged,
only the thing calling it moved.

**Seam 2 -- message history.** Resume needs the framework to continue a
conversation rather than restart it, and `message_history=` is the only way in.
So the full history is persisted after every model response, which is the
tightest granularity `after_model_request` allows.

The gap this leaves, stated plainly: a crash *between* a tool committing and the
next model response being persisted loses that tool's return from the history.
On resume the framework re-issues the call, and the only thing standing between
that and a duplicate send is the ledger check in seam 1 -- which is why the key
must be derived from the replayed conversation and not from `run_step`. See
identity.py.

Part A did not have this gap: it persisted each tool result the instant it
happened. The framework will not let us do that, because a `ToolReturnPart` does
not exist as a persistable message until the turn is assembled.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.tools import RunContext

from agent import storage

from .deps import FwDeps
from .identity import idempotency_key

IRREVERSIBLE = frozenset({"send_email"})


def is_irreversible(tool: str) -> bool:
    return tool in IRREVERSIBLE


async def guarded_execute(
    ctx: RunContext[FwDeps], tool: str, args: dict[str, Any], handler
) -> Any:
    """Run one tool body at most once per logical call, forever.

    `handler` is the framework's own continuation; not calling it is how a
    replayed effect is suppressed.
    """
    deps = ctx.deps

    if not is_irreversible(tool):
        result = await handler(args)
        _trace_tool(ctx, tool, args, result, replayed=False)
        return result

    key = idempotency_key(ctx, deps.run_id, tool, args)

    existing = deps.conn.execute(
        "SELECT result FROM effects WHERE idempotency_key = ?", (key,)
    ).fetchone()
    if existing is not None:
        # Already committed by an earlier process. The body must not run again.
        deps.replayed_effects += 1
        _trace_tool(ctx, tool, args, existing["result"], replayed=True, key=key)
        return existing["result"]

    result = await handler(args)

    outcome = storage.commit_email(
        deps.conn, key, deps.run_id, ctx.run_step, args, result=str(result)
    )
    _trace_tool(ctx, tool, args, outcome.result, replayed=outcome.replayed, key=key)
    return outcome.result


def _trace_tool(
    ctx: RunContext[FwDeps],
    tool: str,
    args: dict[str, Any],
    result: Any,
    replayed: bool,
    key: str | None = None,
) -> None:
    ctx.deps.record(
        ctx.run_step,
        "tool_result",
        {
            "tool": tool,
            "args": args,
            "call_id": ctx.tool_call_id,
            "ok": True,
            "replayed": replayed,
            "idempotency_key": key,
            "content": str(result)[:2000],
        },
    )


# ------------------------------------------------------- history persistence


def persist_history(deps: FwDeps, messages: list[Any]) -> None:
    """Store the conversation so a later process can resume it."""
    blob = ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8")
    storage.append_event(
        deps.conn, deps.run_id, len(messages), "history", {"messages_json": blob}
    )


def load_history(conn, run_id: str) -> list[Any] | None:
    """The most recent persisted conversation for `run_id`, or None."""
    latest = None
    for event in storage.iter_events(conn, run_id):
        if event["kind"] == "history":
            latest = event["payload"]["messages_json"]
    if latest is None:
        return None
    return ModelMessagesTypeAdapter.validate_json(latest)
