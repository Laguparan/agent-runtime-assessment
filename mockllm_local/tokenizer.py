"""Deterministic token counter.

The real `mockllm/tokenizer.py` is the single source of truth for the 8,000
token ceiling in R3. This is a stand-in with the same contract: same input
always yields the same count, no model download, no network.

It is a GPT-2-style greedy splitter (words carry their leading space,
long words subdivide, punctuation is per-character). It will NOT agree
digit-for-digit with the real tokenizer, so treat the 8k ceiling as a soft
target locally and leave headroom.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

# Roughly the GPT-2 pre-tokenizer: contractions, then space-prefixed words,
# then space-prefixed digit runs, then punctuation runs, then whitespace.
_PIECE_RE = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)      # common contractions
      | \ ?[A-Za-z]+               # word, optionally carrying one leading space
      | \ ?\d+                     # digit run
      | \ ?[^\sA-Za-z\d]+          # punctuation / symbol run
      | \s+                        # whitespace run
    """,
    re.VERBOSE,
)

# A piece longer than this is assumed to break into multiple BPE tokens.
_CHARS_PER_SUBTOKEN = 4

# Per-message framing overhead the real APIs also charge for (role, delimiters).
_MESSAGE_OVERHEAD_TOKENS = 4


def count_tokens(text: Any) -> int:
    """Count tokens in `text`. Non-strings are stringified first."""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0

    total = 0
    for match in _PIECE_RE.finditer(text):
        piece = match.group()
        if piece.isspace():
            # Whitespace runs are cheap but not free; newlines dominate the cost.
            total += max(1, piece.count("\n"))
            continue
        total += max(1, math.ceil(len(piece.strip()) / _CHARS_PER_SUBTOKEN))
    return total


def count_message_tokens(messages: Iterable[dict]) -> int:
    """Count tokens for a conversation, including per-message framing overhead.

    Accepts both wire shapes: `content` may be a plain string or a list of
    Anthropic-style content blocks, and OpenAI-style `tool_calls` are counted.
    """
    total = 0
    for message in messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        total += count_tokens(message.get("role", ""))
        total += _count_content(message.get("content"))

        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            total += count_tokens(function.get("name", ""))
            total += count_tokens(function.get("arguments", ""))
    return total


def _count_content(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return count_tokens(content)
    if isinstance(content, list):
        return sum(_count_block(block) for block in content)
    return count_tokens(json.dumps(content, sort_keys=True))


def _count_block(block: Any) -> int:
    if not isinstance(block, dict):
        return count_tokens(block)

    block_type = block.get("type")
    if block_type == "text":
        return count_tokens(block.get("text", ""))
    if block_type == "tool_use":
        return count_tokens(block.get("name", "")) + count_tokens(
            json.dumps(block.get("input", {}), sort_keys=True)
        )
    if block_type == "tool_result":
        return _count_content(block.get("content"))
    return count_tokens(json.dumps(block, sort_keys=True))


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or "The quick brown fox jumps over the lazy dog."
    print(f"{count_tokens(sample)} tokens: {sample!r}")
