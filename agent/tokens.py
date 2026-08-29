"""Tokenizer resolution, in one place.

Order matters. The assessment's own tokenizer is authoritative for R3's 8,000
ceiling; ours is a stand-in that will not agree with it digit for digit, which
is why `config.COMPACT_AT` leaves 2,000 tokens of headroom.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCE = "estimate"

try:  # 1. the real thing, if it is ever provided
    from mockllm.tokenizer import count_tokens as _count_tokens  # type: ignore

    SOURCE = "mockllm"
except ImportError:
    try:  # 2. our local stand-in
        from mockllm_local.tokenizer import count_tokens as _count_tokens

        SOURCE = "mockllm_local"
    except ImportError:  # 3. last resort
        def _count_tokens(text):  # type: ignore[misc]
            return len(str(text)) // 4


def count_tokens(text) -> int:
    return _count_tokens(text)


def count_messages(messages) -> int:
    """Token cost of a message list, with per-message framing overhead."""
    total = 0
    for message in messages:
        total += 4 + count_tokens(message.get("role", ""))
        content = message.get("content")
        if isinstance(content, list):
            total += sum(count_tokens(str(block)) for block in content)
        else:
            total += count_tokens(content)
    return total
