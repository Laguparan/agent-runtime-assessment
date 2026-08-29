"""R3 under the framework: the 8,000 token ceiling.

Two halves, and the framework provides one of them.

**Provided.** `UsageLimits` is a real budget primitive. `request_limit` is R5's
step ceiling, `total_tokens_limit` and `cost_limit` are its cost budget, and
`per_request_input_tokens_limit` is a hard guard on the size of any single
request. Using them means the ceiling is enforced by the framework rather than
by a check we remembered to write, which is strictly better than Part A.

**Not provided.** Nothing compacts. `UsageLimits` raises when the budget is
exceeded; it does not make the transcript smaller so the run can continue. The
compaction strategy itself is ours either way, so the only question was where to
hang it, and `ProcessHistory` is the right seam: it rewrites the message list on
its way to the model and leaves the durable history untouched.

That happens to preserve the property Part A was careful about -- compaction is
a *view*, so no later turn reasons against history an earlier compaction
destroyed, and what we persist for resume is always complete.

The strategy is the one defended in Part A's DECISIONS.md, ported to the
framework's message types: digest tool output first, everywhere, before
touching a word the model reasoned. See agent/memory.py for why that ordering
is what makes the turn-3-fact-at-turn-40 task pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext

from agent import config
from agent.tokens import count_tokens

from .deps import FwDeps

RECENT_MESSAGES = 6
ANCHOR_MESSAGES = 6
DIGEST_CHARS = config.TOOL_RESULT_DIGEST_CHARS


def _size(messages: list[ModelMessage]) -> int:
    return count_tokens(repr(messages))


def _digest_return(part: ToolReturnPart) -> ToolReturnPart:
    content = str(part.content)
    if len(content) <= DIGEST_CHARS:
        return part
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return dataclasses.replace(
        part,
        content=(
            f"{content[:DIGEST_CHARS]}\n[compacted: {len(content)} chars total, "
            f"sha256:{fingerprint}. Re-read the source if you need the rest.]"
        ),
    )


def _digest_message(message: ModelMessage) -> ModelMessage:
    if not isinstance(message, ModelRequest):
        return message
    parts = [
        _digest_return(part) if isinstance(part, ToolReturnPart) else part
        for part in message.parts
    ]
    return dataclasses.replace(message, parts=parts)


def _shrink_text(message: ModelMessage) -> ModelMessage:
    if not isinstance(message, ModelResponse):
        return message
    parts = []
    for part in message.parts:
        if isinstance(part, TextPart) and len(part.content) > DIGEST_CHARS:
            parts.append(
                dataclasses.replace(
                    part, content=part.content[:DIGEST_CHARS] + "\n[reasoning compacted]"
                )
            )
        else:
            parts.append(part)
    return dataclasses.replace(message, parts=parts)


async def compact_history(
    ctx: RunContext[FwDeps], messages: list[ModelMessage]
) -> list[ModelMessage]:
    """Shrink the outgoing message list, leaving the stored history alone."""
    before = _size(messages)
    if before < config.COMPACT_AT or len(messages) <= RECENT_MESSAGES:
        return messages

    recent = messages[-RECENT_MESSAGES:]
    older = messages[: len(messages) - RECENT_MESSAGES]

    # Pass 1: digest every tool return outside the recent window.
    older = [_digest_message(m) for m in older]
    working = older + recent

    # Pass 2: trim model prose outside the anchor.
    if _size(working) >= config.COMPACT_AT:
        anchor, tail = older[:ANCHOR_MESSAGES], older[ANCHOR_MESSAGES:]
        working = anchor + [_shrink_text(m) for m in tail] + recent

    after = _size(working)
    if isinstance(ctx.deps, FwDeps):
        ctx.deps.compactions += 1
        ctx.deps.record(
            ctx.run_step,
            "compaction",
            {"before": before, "after": after, "messages": len(working)},
        )
    return working
