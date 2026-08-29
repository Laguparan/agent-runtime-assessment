"""F2: giving a tool call a stable identity when you do not own the loop.

Part A keyed an irreversible effect on `(run_id, step, index, tool, args)`,
where `step` was the runtime's own loop counter and therefore stable across a
crash: resume replayed the recorded model responses, so the same logical send
landed on the same step every time.

Under the framework there is no such counter available to us.

`RunContext.run_step` looks like the obvious replacement and is a trap. It
counts model requests *within the current process*, so it restarts at 1 on
resume. Measured, replaying a conversation that had already made four calls:

    fresh run    read_file(1) read_file(2) send_email(3) send_email(4)
    resumed run  ...          ...          send_email(1) send_email(2)

Keying on it would give the resumed `send_email` the key of the *first* one, so
the second send would be suppressed as a duplicate -- exactly-once failing in
the "never" direction, which is the failure the chaos harness caught in Part A.

What is stable is the conversation itself, because the conversation is what we
persist and replay. The number of `ToolReturnPart`s already present in
`ctx.messages` is a pure function of the replayed history: identical before the
crash and after it. That is the anchor used here.

Parallel calls in one response share that count, so the position of this call
within the current response disambiguates them -- the same role `index` played
in Part A.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import RunContext


def completed_tool_calls(ctx: RunContext[Any]) -> int:
    """How many tool calls this conversation has already returned results for."""
    return sum(
        1
        for message in ctx.messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


def call_index_in_turn(ctx: RunContext[Any]) -> int:
    """Position of this call among the tool calls of the current model response.

    Falls back to 0 when the call cannot be located, which is safe: it only
    matters for distinguishing *parallel* calls, and a single call is always 0.
    """
    for message in reversed(ctx.messages):
        if not isinstance(message, ModelResponse):
            continue
        calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
        for index, part in enumerate(calls):
            if part.tool_call_id == ctx.tool_call_id:
                return index
        break
    return 0


def idempotency_key(ctx: RunContext[Any], run_id: str, tool: str, args: dict[str, Any]) -> str:
    """Stable identity for one logical effect, across crashes and resumes.

    Deliberately not derived from `tool_call_id`: S9 reuses one id across four
    distinct calls including two different sends, so keying on it collapses
    them. The framework does not deduplicate those ids for us either -- it
    passes through whatever the model emitted.
    """
    material = json.dumps(
        {
            "run": run_id,
            "position": completed_tool_calls(ctx),
            "index": call_index_in_turn(ctx),
            "tool": tool,
            "args": args,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
