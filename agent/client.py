"""Resilient client for the mock model server.

Built on `http.client` rather than `urllib` for one reason: urllib hides the
difference between a response that ended early and one that ended. S5 (RST at a
random byte offset) and S12 (clean FIN short of Content-Length) both have to be
distinguishable from a real answer, and `IncompleteRead` is how that shows up.

Failure classification is the whole job here:

  retryable    connection reset, incomplete read, unparseable body,
               429 / 529 / 500 / 502 / 503
  fatal        every other 4xx -- retrying a 400 just spends the budget

Retry-After is honoured but clamped: a hostile server can send
`Retry-After: 86400` and a client that trusts it has hung for a day.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import random
import time
import urllib.parse
from typing import Any

from . import config


class ModelUnavailable(Exception):
    """Every attempt failed. The loop terminates gracefully on this."""


class ModelRefused(Exception):
    """A non-retryable HTTP error. Retrying would not help."""


@dataclasses.dataclass
class ToolCall:
    id: str
    name: str
    index: int
    args: dict[str, Any] | None      # None when the arguments would not parse
    raw: str | None = None           # the unparseable text, for the error message
    parse_error: str | None = None


@dataclasses.dataclass
class ModelResponse:
    text: str
    calls: list[ToolCall]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    attempts: int
    raw: dict[str, Any]

    @property
    def wants_tools(self) -> bool:
        return bool(self.calls)


RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 529})


def _sleep_for(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), config.MAX_RETRY_AFTER)
        except ValueError:
            pass
    # Exponential with jitter, so a chaos loop of clients does not resonate.
    return min(config.MAX_BACKOFF, (2**attempt) * 0.25) * (0.5 + random.random() / 2)


def parse_response(body: dict[str, Any]) -> ModelResponse:
    """Normalise an Anthropic-shaped body into a ModelResponse.

    A `tool_use` block whose `input` is a string is a malformed call (S2). It is
    recorded as a call with `args=None` rather than discarded -- the model made
    the call, so the transcript owes it a result, and dropping it would leave a
    dangling tool_use that corrupts the next turn.
    """
    text_parts: list[str] = []
    calls: list[ToolCall] = []

    for index, block in enumerate(body.get("content") or []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            raw_input = block.get("input")
            call = ToolCall(
                id=str(block.get("id") or f"call_{index}"),
                name=str(block.get("name") or ""),
                index=len(calls),
                args=None,
            )
            if isinstance(raw_input, dict):
                call.args = raw_input
            elif isinstance(raw_input, str):
                try:
                    parsed = json.loads(raw_input)
                    call.args = parsed if isinstance(parsed, dict) else None
                    if call.args is None:
                        call.raw = raw_input
                        call.parse_error = "arguments parsed to a non-object"
                except json.JSONDecodeError as exc:
                    call.raw = raw_input
                    call.parse_error = str(exc)
            else:
                call.raw = json.dumps(raw_input, default=str)
                call.parse_error = f"arguments were {type(raw_input).__name__}, not an object"
            calls.append(call)

    usage = body.get("usage") or {}
    return ModelResponse(
        text="\n".join(p for p in text_parts if p),
        calls=calls,
        stop_reason=str(body.get("stop_reason") or "end_turn"),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        attempts=1,
        raw=body,
    )


class ModelClient:
    def __init__(
        self,
        base_url: str | None = None,
        session: str | None = None,
        scenario: str | None = None,
        on_retry=None,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url or config.MOCK_BASE_URL)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8000
        self.session = session
        self.scenario = scenario
        # Called with (attempt, reason, wait) so the loop can record every retry
        # in the trace -- F4 asks for exactly this, and R6 needs it too.
        self.on_retry = on_retry

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        step: int | None = None,
    ) -> ModelResponse:
        payload = json.dumps(
            {
                "model": "mock-model",
                "max_tokens": 2048,
                "system": config.SYSTEM_PROMPT,
                "messages": _to_wire(messages),
                "tools": tools,
            }
        )
        headers = {"Content-Type": "application/json"}
        if self.session:
            headers["X-Mock-Session"] = self.session
        if self.scenario:
            headers["X-Mock-Scenario"] = self.scenario

        last_reason = "no attempt was made"
        for attempt in range(config.MAX_ATTEMPTS):
            if step is not None:
                # Lets the mock serve a turn as a pure function of (step, attempt),
                # so a crash-and-resume re-requesting this step sees the same
                # turn it saw before. Ignored by a server that does not know it.
                headers["X-Mock-Step"] = str(step)
                headers["X-Mock-Attempt"] = str(attempt)
            reason, retry_after, response = self._attempt(payload, headers)
            if response is not None:
                response.attempts = attempt + 1
                return response

            last_reason = reason
            if attempt == config.MAX_ATTEMPTS - 1:
                break
            wait = _sleep_for(attempt, retry_after)
            if self.on_retry:
                self.on_retry(attempt + 1, reason, wait)
            time.sleep(wait)

        raise ModelUnavailable(
            f"model server did not answer in {config.MAX_ATTEMPTS} attempts; "
            f"last failure: {last_reason}"
        )

    def _attempt(
        self, payload: str, headers: dict[str, str]
    ) -> tuple[str, str | None, ModelResponse | None]:
        """One request. Returns (reason, retry_after, response-or-None)."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=config.REQUEST_TIMEOUT)
        try:
            conn.request("POST", config.MESSAGES_PATH, payload, headers)
            raw_response = conn.getresponse()
            status = raw_response.status
            retry_after = raw_response.getheader("Retry-After")

            try:
                raw_body = raw_response.read()
            except http.client.IncompleteRead as exc:
                # S5 and S12 both land here. Partial bytes are not a response.
                return (
                    f"incomplete read ({len(exc.partial)} of "
                    f"{len(exc.partial) + (exc.expected or 0)} bytes)",
                    retry_after,
                    None,
                )

            if status in RETRYABLE_STATUS:
                return (f"HTTP {status}", retry_after, None)
            if status >= 400:
                raise ModelRefused(f"HTTP {status}: {raw_body[:200].decode('utf-8', 'replace')}")

            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return (f"unparseable response body: {exc}", retry_after, None)

            return ("ok", None, parse_response(body))

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            return (f"connection reset ({exc.__class__.__name__})", None, None)
        except (http.client.HTTPException, TimeoutError, OSError) as exc:
            return (f"{exc.__class__.__name__}: {exc}", None, None)
        finally:
            conn.close()


def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the runtime's flat transcript into Messages-API content blocks.

    The runtime keeps a flat internal shape because compaction and the event log
    are far easier to reason about that way. Translation happens once, here, at
    the boundary. `system` is carried in its own top-level field, and keys
    starting with `_` are runtime bookkeeping that never goes on the wire.

    Note on S9: tool_result blocks reference `tool_use_id`, and the model reuses
    ids. The wire is genuinely ambiguous when that happens; the runtime is not,
    because internally results are keyed on (step, index). See DECISIONS.md.
    """
    wire: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue

        if role == "tool":
            wire.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_use_id", ""),
                            "content": str(message.get("content", "")),
                            "is_error": bool(message.get("is_error")),
                        }
                    ],
                }
            )
            continue

        blocks: list[dict[str, Any]] = []
        if message.get("content"):
            blocks.append({"type": "text", "text": str(message["content"])})
        for call in message.get("tool_calls") or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "input": call.get("args") if call.get("args") is not None else call.get("raw", ""),
                }
            )
        wire.append({"role": role or "user", "content": blocks or [{"type": "text", "text": ""}]})

    return wire


def estimated_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1000 * config.COST_PER_1K_INPUT
        + output_tokens / 1000 * config.COST_PER_1K_OUTPUT
    )
