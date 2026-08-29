"""Render a scenario Turn onto the wire, in either API shape.

Two shapes are supported from one scenario script:

  * `/v1/messages`         Anthropic Messages shape -- what the brief specifies.
  * `/v1/chat/completions` OpenAI Chat Completions shape -- so an off-the-shelf
                           OpenAI-compatible client can point at this server
                           unmodified for Part B.

Malformed arguments (S2) look different in each shape. OpenAI carries tool
arguments as a JSON *string*, so broken JSON rides along untouched. Anthropic
normally carries them as an object; here a malformed call emits `input` as a
*string* holding the raw broken text instead, which is the non-streaming
equivalent of the partial `input_json_delta` a real stream would produce.

The envelope stays parseable either way. That is deliberate: a client cannot
recover from a response it cannot parse at all, and "unparseable envelope" is
already covered by S5 and S12.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from . import tokenizer
from .scenario import Scenario, Turn

SHAPE_ANTHROPIC = "anthropic"
SHAPE_OPENAI = "openai"

# S8: how much bigger each successive response gets.
_GROWTH_BASE = 2
_GROWTH_MAX_REPEATS = 512  # keeps a runaway agent from OOMing the server


def render(
    turn: Turn,
    scenario: Scenario,
    shape: str,
    model: str,
    request_messages: list[dict],
    cursor: int = 0,
) -> bytes:
    """Serialise `turn` to response bytes in the requested wire `shape`."""
    text = _grown_text(turn, cursor)
    input_tokens = tokenizer.count_message_tokens(request_messages)
    output_tokens = tokenizer.count_tokens(text) + sum(
        tokenizer.count_tokens(call.raw_arguments or json.dumps(call.input))
        for call in turn.tool_calls
    )

    if shape == SHAPE_ANTHROPIC:
        body = _anthropic_body(turn, scenario, text, model, input_tokens, output_tokens)
    else:
        body = _openai_body(turn, scenario, text, model, input_tokens, output_tokens)

    return json.dumps(body).encode("utf-8")


def _grown_text(turn: Turn, cursor: int) -> str:
    """S8: double the response text with every turn of the conversation.

    Keyed on the conversation cursor rather than on how far past the end of the
    script we are, so growth is visible inside a scripted scenario too -- not
    only once a `repeat_last` turn starts looping.
    """
    if not turn.grow or cursor <= 0:
        return turn.text
    repeats = min(_GROWTH_BASE ** cursor, _GROWTH_MAX_REPEATS)
    return " ".join([turn.text.strip()] * repeats)


def _anthropic_body(
    turn: Turn,
    scenario: Scenario,
    text: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    if text:
        content.append({"type": "text", "text": text})

    for call in turn.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                # A well-formed call carries an object. A malformed one carries
                # the raw broken text as a string, and the client has to notice.
                "input": call.input if call.raw_arguments is None else call.raw_arguments,
            }
        )

    body = {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": turn.resolved_stop_reason(),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        # Not part of the real API. Left in deliberately so a trace makes it
        # obvious which scripted turn produced which response.
        "_mock": {"scenario": scenario.id},
    }
    return body


_OPENAI_FINISH_REASON = {
    "tool_use": "tool_calls",
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _openai_body(
    turn: Turn,
    scenario: Scenario,
    text: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                # OpenAI arguments are already a string, so malformed JSON
                # round-trips through normal escaping. No sentinel needed.
                "arguments": (
                    call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.input)
                ),
            },
        }
        for call in turn.tool_calls
    ]

    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    body = {
        "id": f"chatcmpl_{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": _OPENAI_FINISH_REASON.get(
                    turn.resolved_stop_reason(), "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "_mock": {"scenario": scenario.id},
    }
    return body


def error_body(shape: str, status: int, message: str) -> bytes:
    """An error payload in the shape the client expects."""
    if shape == SHAPE_ANTHROPIC:
        error_type = {
            429: "rate_limit_error",
            529: "overloaded_error",
            400: "invalid_request_error",
            404: "not_found_error",
        }.get(status, "api_error")
        body = {"type": "error", "error": {"type": error_type, "message": message}}
    else:
        body = {
            "error": {
                "message": message,
                "type": "rate_limit_error" if status == 429 else "server_error",
                "code": status,
            }
        }
    return json.dumps(body).encode("utf-8")
