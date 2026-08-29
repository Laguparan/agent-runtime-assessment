"""F4: what the framework reports about a run, and what it hides.

The brief asks for four numbers per run: total tokens, step count, every retry
the framework performed on our behalf, and every error it swallowed. Two come
free and two do not.

**Free.** `result.usage` is a `RunUsage` with `input_tokens`, `output_tokens`,
`requests` and `tool_calls`. Steps are `usage.requests`.

**Not free -- retries.** `usage.requests` counts requests the *framework* made,
not requests that reached the server. The Anthropic SDK retries beneath it, and
those retries are invisible. Measured against the mock:

    S5 (two mid-body connection resets)   server answered 4, usage.requests = 2
    S6 (429 with Retry-After, then 529)   server answered 4, usage.requests = 2

So half the traffic in both scenarios is unaccounted for at the framework level.
The only place to see it is below the framework, at the transport: this module
installs an httpx event hook and counts responses there. `transport_retries` is
the difference between what the wire saw and what the framework admits to.

That is the honest answer to "every retry the framework performed on your
behalf": it will not tell you, and you have to instrument the layer underneath
it to find out.

**Not free -- swallowed errors.** A malformed tool call (S2) becomes a
`ModelRetry` fed back to the model and never surfaces to the caller. The
`on_tool_validate_error` and `on_tool_execute_error` hooks are where those
become visible; without them a run reports success having quietly absorbed
three failures.
"""

from __future__ import annotations

from typing import Any

import httpx2

from .deps import FwDeps


def counting_http_client(deps: FwDeps) -> httpx2.AsyncClient:
    """An httpx client that counts every response the wire actually returned.

    This sits below the Anthropic SDK, so it sees the retried requests the
    framework's own usage accounting does not.
    """

    async def on_request(request: httpx2.Request) -> None:
        # Part A set these directly because it owned the HTTP call. Here the
        # framework builds the request and offers no hook to add a header to it,
        # so the only reachable place is the transport it was handed.
        #
        # X-Mock-Session keeps each run on its own scenario cursor -- without it
        # every run sharing a task string shares a cursor, and the second run
        # starts partway through the script.
        #
        # X-Mock-Step / X-Mock-Attempt put the mock in addressed mode, so a
        # resumed run re-requesting a step sees the turn it saw before. The
        # attempt counter increments here rather than in the framework, which
        # means SDK-level retries advance it exactly as a Part A retry would.
        request.headers["X-Mock-Session"] = deps.run_id
        request.headers["X-Mock-Step"] = str(deps.mock_step)
        request.headers["X-Mock-Attempt"] = str(deps.mock_attempt)
        deps.mock_attempt += 1

    async def on_response(response: httpx2.Response) -> None:
        deps.http_requests += 1
        if response.status_code >= 400:
            # A retryable status the SDK will handle without telling anyone.
            deps.record_swallowed(
                "transport",
                RuntimeError(f"HTTP {response.status_code}"),
                detail=str(response.headers.get("retry-after", "")),
            )

    return httpx2.AsyncClient(
        event_hooks={"request": [on_request], "response": [on_response]}, timeout=30.0
    )


def report(deps: FwDeps, usage: Any, output: Any, stop_reason: str, detail: str) -> dict[str, Any]:
    """The per-run extraction the brief asks for, plus what it took to get it."""
    framework_requests = int(getattr(usage, "requests", 0) or 0)
    deps.transport_retries = max(0, deps.http_requests - framework_requests)

    return {
        "run_id": deps.run_id,
        "stop_reason": stop_reason,
        "detail": detail,
        # --- reported by the framework ---
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "steps": framework_requests,
        "tool_calls": int(getattr(usage, "tool_calls", 0) or 0),
        # --- had to be instrumented ---
        "http_requests": deps.http_requests,
        "transport_retries": deps.transport_retries,
        "tool_retries": deps.tool_retries,
        "swallowed_errors": deps.swallowed_errors,
        # --- ours ---
        "policy_denials": deps.denials,
        "replayed_effects": deps.replayed_effects,
        "compactions": deps.compactions,
        "output": str(output)[:500] if output is not None else None,
    }


def summary(data: dict[str, Any], emails: int) -> str:
    return (
        f"{data['run_id']}: {data['stop_reason']} after {data['steps']} steps "
        f"({data['detail']}). tokens in/out {data['input_tokens']}/{data['output_tokens']}, "
        f"{data['tool_calls']} tool calls, {data['policy_denials']} denied, "
        f"{emails} emails.\n"
        f"  framework reported {data['steps']} requests; the wire saw "
        f"{data['http_requests']} ({data['transport_retries']} retried beneath it).\n"
        f"  {data['tool_retries']} tool retries, "
        f"{len(data['swallowed_errors'])} errors swallowed."
    )
