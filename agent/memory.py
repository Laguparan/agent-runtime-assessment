"""R3: keeping the transcript under the 8,000 token ceiling.

Strategy: **compact by kind first, by position second.**

Four passes, escalating, stopping as soon as the transcript fits:

    1. digest every tool result outside the recent window
    2. trim assistant prose outside the anchor
    3. drop the non-anchor middle, leaving a visible marker
    4. trim the recent window down to the last two messages

The rule that does the work is that *tool output is compressible and model
reasoning is not*. A tool result is a fact the model has already read and
reacted to; the reaction is in the assistant message that follows it, so the
bulk can be replaced by a digest without losing the conclusion drawn from it.

Ordering the passes by kind rather than by position is the whole reason the
turn-3-fact-at-turn-40 task passes. An earlier version of this file anchored
the first N messages and digested everything after them, which failed that
test outright: turn 3 of a conversation is around message eight once each turn
contributes an assistant message and a tool result, so the fact fell one slot
outside the anchor and was trimmed. Digesting all tool output first removes far
more tokens than trimming prose ever will, and it removes them from the part of
the transcript nobody needs verbatim.

Defended against the obvious alternative -- recursive LLM summarisation of the
middle -- in DECISIONS.md. Short version: summarisation costs a model call per
compaction on a server that returns 429s, and it is nondeterministic, which
would break R6's replay guarantee outright.
"""

from __future__ import annotations

import hashlib
from typing import Any

from . import config
from .tokens import count_messages, count_tokens


class ContextExhausted(Exception):
    """Compaction ran out of things to compress and the ceiling still stands."""


def _digest(message: dict[str, Any]) -> dict[str, Any]:
    """Replace a tool result with a short, honest summary of what it was."""
    content = str(message.get("content", ""))
    if len(content) <= config.TOOL_RESULT_DIGEST_CHARS:
        return message

    head = content[: config.TOOL_RESULT_DIGEST_CHARS]
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return {
        **message,
        "content": (
            f"{head}\n[compacted: {len(content)} chars total, sha256:{fingerprint}. "
            f"Re-read the source if you need the rest.]"
        ),
        "_compacted": True,
    }


def _shrink_assistant(message: dict[str, Any]) -> dict[str, Any]:
    """Second-pass fallback: trim assistant prose. Tool calls are never touched."""
    content = str(message.get("content", ""))
    if len(content) <= config.TOOL_RESULT_DIGEST_CHARS or message.get("_compacted"):
        return message
    return {
        **message,
        "content": content[: config.TOOL_RESULT_DIGEST_CHARS] + "\n[reasoning compacted]",
        "_compacted": True,
    }


def compact(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (messages, stats). Raises ContextExhausted if the ceiling holds."""
    before = count_messages(messages)
    stats = {"before": before, "after": before, "passes": 0, "compacted": False}

    if before < config.COMPACT_AT:
        return messages, stats

    head = messages[:2]                                  # system + original task
    body = messages[2:]
    recent = body[-config.RECENT_MESSAGES :] if len(body) > config.RECENT_MESSAGES else []
    older = body[: len(body) - len(recent)]

    # Pass 1: digest EVERY tool result outside the recent window, wherever it
    # sits. Ordering the passes by kind rather than by position is what makes
    # the turn-3 fact survive: a positional anchor only protects a fact that
    # happens to fall inside it, and turn 3 of a real conversation is already
    # message eight or so. Tool output is where the bulk always is, so digest
    # all of it before touching a single word the model reasoned.
    older = [_digest(m) if m.get("role") == "tool" else m for m in older]
    working = head + older + recent
    stats["passes"] = 1

    # From here on the anchor matters: the earliest reasoning is the most likely
    # to hold a durable fact that later turns depend on.
    anchor = older[: config.ANCHOR_MESSAGES]
    tail = older[config.ANCHOR_MESSAGES :]

    # Pass 2: trim assistant prose outside the anchor.
    if count_messages(working) >= config.COMPACT_AT:
        tail = [_shrink_assistant(m) if m.get("role") == "assistant" else m for m in tail]
        working = head + anchor + tail + recent
        stats["passes"] = 2

    # Pass 3: collapse the non-anchor tool output into a single marker, and keep
    # every assistant message. Dropping the tail wholesale is the obvious move
    # and it is wrong: by this point the tool results are already digested to a
    # couple of hundred characters each while the reasoning is a sentence each,
    # so the tool results are still nearly all of the weight -- and the
    # reasoning is where a fact stated once at turn 3 lives. Collapsing thirty
    # digests recovers far more than deleting thirty sentences, and costs
    # nothing that a later turn can need verbatim.
    dropped = sum(1 for m in tail if m.get("role") == "tool")
    if count_messages(working) >= config.COMPACT_AT and dropped:
        tail = [m for m in tail if m.get("role") != "tool"] + [
            {
                "role": "tool",
                "name": "context",
                "content": f"[{dropped} earlier tool results dropped to stay within "
                           f"the token budget. Re-read a source if you need it "
                           f"again; the reasoning about them is preserved above.]",
                "_compacted": True,
            }
        ]
        working = head + anchor + tail + recent
        stats["passes"] = 3

    # Pass 4: the recent window is the last thing touched, because it is the
    # turn the model is actually working on. Under S8 a single recent message
    # can be larger than the whole budget, so it has to be reachable.
    if count_messages(working) >= config.COMPACT_AT and len(recent) > config.KEEP_VERBATIM:
        keep = recent[-config.KEEP_VERBATIM :]
        trimmed = [
            _digest(m) if m.get("role") == "tool" else _shrink_assistant(m)
            for m in recent[: -config.KEEP_VERBATIM]
        ]
        working = head + anchor + tail + trimmed + keep
        stats["passes"] = 4

    after = count_messages(working)
    stats.update(after=after, compacted=True)

    if after >= config.TOKEN_CEILING:
        raise ContextExhausted(
            f"context is {after} tokens after {stats['passes']} compaction passes, "
            f"ceiling is {config.TOKEN_CEILING}. Nothing further can be dropped "
            f"without discarding the anchor or the current turn."
        )
    return working, stats


def would_exceed(messages: list[dict[str, Any]]) -> bool:
    return count_messages(messages) >= config.TOKEN_CEILING


# Kept for the older call sites and the eval suite.
def compact_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compact(messages)[0]


__all__ = [
    "ContextExhausted",
    "compact",
    "compact_history",
    "count_tokens",
    "would_exceed",
]
