"""Assembles the Pydantic AI agent: model, tools, hooks, and budgets.

F1 parity, honestly labelled. Where the framework has a primitive it is used;
where it does not, that is said rather than quietly worked around.

    R1  survive S1-S12          framework, mostly. Tool validation, retries and
                                transcript assembly are all its own. The gaps
                                are listed under no-progress below.
    R3  8k ceiling              SPLIT. UsageLimits.per_request_input_tokens_limit
                                enforces it; compaction to stay under it is ours,
                                hung on ProcessHistory.
    R5  step ceiling            framework: UsageLimits.request_limit.
    R5  token / cost budget     framework: total_tokens_limit, cost_limit.
    R5  per-tool timeout        framework: Agent(tool_timeout=...).
    R5  no-progress detection   NOT PROVIDED. Hand-rolled in before_model_request.
                                Without it S4 runs to the request ceiling -- 50
                                requests by default, measured -- instead of
                                stopping after three identical calls.
"""

from __future__ import annotations

import dataclasses
import json

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import Hooks, ProcessHistory
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import UsageLimits

from agent import config
from agent import tools as part_a_tools

from .boundary import Refusal, enforce
from .compaction import compact_history
from .deps import FwDeps
from .durability import guarded_execute, persist_history
from .telemetry import counting_http_client

# S4 emits the same call forever. The framework has no notion of progress, so
# this is the hand-rolled half of R5.
NO_PROGRESS_LIMIT = config.NO_PROGRESS_LIMIT


class NoProgress(Exception):
    """Raised to stop a run that is repeating itself."""


def usage_limits() -> UsageLimits:
    return UsageLimits(
        request_limit=config.MAX_STEPS,
        per_request_input_tokens_limit=config.TOKEN_CEILING,
        # The mock reports no cost, so a cost_limit here would never fire and
        # claiming it as coverage would be dishonest.
        cost_limit=None,
    )


def build_agent(deps: FwDeps, scenario: str | None) -> Agent:
    """Construct the agent. Every seam we use is wired here and nowhere else."""
    model_name = f"mock-{scenario.lower()}" if scenario else "mock-model"
    model = AnthropicModel(
        model_name,
        provider=AnthropicProvider(
            base_url=config.MOCK_BASE_URL,
            api_key="not-needed",
            http_client=counting_http_client(deps),
        ),
    )

    hooks: Hooks = Hooks()
    _register_tool_hooks(hooks)
    _register_checkpoint_hook(hooks)
    _register_progress_hook(hooks)
    _register_output_cap(hooks)
    _register_error_hooks(hooks)

    agent: Agent = Agent(
        model,
        deps_type=FwDeps,
        instructions=config.SYSTEM_PROMPT,
        # S2 sends three malformed calls in a row. The default budget is 1, and
        # measured, the run dies on the second. This is the framework's own
        # malformed-argument recovery; it just needs headroom to use it.
        retries=6,
        # S10 hangs one of three parallel calls. This is the framework's timeout.
        tool_timeout=config.TOOL_TIMEOUT,
        capabilities=[hooks, ProcessHistory(compact_history)],
    )
    _register_tools(agent)
    return agent


# ------------------------------------------------------------------- tooling


def _ctx_for(ctx: RunContext, index: int = 0) -> part_a_tools.ToolContext:
    """Adapt the framework's context to Part A's tool signature, unchanged."""
    return part_a_tools.ToolContext(
        ctx.deps.run_id, ctx.run_step, index, ctx.deps.policy, ctx.deps.conn
    )


def _register_tools(agent: Agent) -> None:
    """The same five tools, with the same bodies as Part A.

    The bodies are imported, not rewritten. That is the exit-cost answer in F5:
    the framework owns orchestration, and the code that does the actual work
    does not know it exists.

    send_email is the exception: its body is a no-op string here, because the
    ledger commit happens in the tool_execute hook where the idempotency key
    can be derived. See durability.py.
    """

    @agent.tool
    def read_file(ctx: RunContext, path: str) -> str:
        """Read a UTF-8 text file from the workspace."""
        return part_a_tools.execute(_ctx_for(ctx), "read_file", {"path": path}).content

    @agent.tool
    def write_file(ctx: RunContext, path: str, content: str) -> str:
        """Write a UTF-8 text file into the workspace."""
        return part_a_tools.execute(
            _ctx_for(ctx), "write_file", {"path": path, "content": content}
        ).content

    @agent.tool
    def run_python(ctx: RunContext, code: str) -> str:
        """Run a Python snippet in a subprocess with no network and a time limit."""
        return part_a_tools.execute(_ctx_for(ctx), "run_python", {"code": code}).content

    @agent.tool
    def http_get(ctx: RunContext, url: str) -> str:
        """HTTP GET an allow-listed URL."""
        return part_a_tools.execute(_ctx_for(ctx), "http_get", {"url": url}).content

    @agent.tool
    def send_email(ctx: RunContext, to: str, subject: str, body: str) -> str:
        """Send an email. Irreversible."""
        return f"Email sent to {to} with subject {subject!r}."


# --------------------------------------------------------------------- hooks


def _register_tool_hooks(hooks: Hooks) -> None:
    @hooks.on.before_tool_execute
    async def check_policy(ctx: RunContext, *, call, tool_def, args):
        """F3: the capability check, before any tool body runs."""
        try:
            enforce(ctx, call.tool_name, dict(args or {}))
        except Refusal as exc:
            raise ModelRetry(str(exc)) from exc
        return args

    @hooks.on.tool_execute
    async def ledger(ctx: RunContext, *, call, tool_def, args, handler):
        """F2: at most one execution per logical call, forever."""
        return await guarded_execute(ctx, call.tool_name, dict(args or {}), handler)


def _register_checkpoint_hook(hooks: Hooks) -> None:
    @hooks.on.before_model_request
    async def checkpoint(ctx: RunContext, request_context):
        """F2: persist the conversation before every model request.

        Persisting only when the run returns -- the obvious place, and where
        this started -- means a killed run has nothing to resume from, so the
        next process starts the conversation over and re-issues every tool call
        it already made. Measured: zero history rows for any killed iteration.

        `before_model_request` is the tightest checkpoint the framework offers.
        By the time it fires for step N+1, ctx.messages holds every response and
        every tool return through step N, so each checkpoint is a complete
        conversation rather than a partial turn.

        Note this is ctx.messages, not request_context.messages: the latter is
        the compacted *view* built by ProcessHistory, and persisting that would
        make compaction permanent and irreversible across a resume.
        """
        persist_history(ctx.deps, list(ctx.messages))

        # Address the mock by conversation position, not by a process-local
        # counter, for the same reason the idempotency key is: run_step restarts
        # at 1 on resume. See identity.py.
        ctx.deps.mock_step = sum(1 for m in ctx.messages if isinstance(m, ModelResponse))
        ctx.deps.mock_attempt = 0
        return request_context


def _register_progress_hook(hooks: Hooks) -> None:
    @hooks.on.before_model_request
    async def stop_if_looping(ctx: RunContext, request_context):
        """R5's missing half: the framework has no notion of progress."""
        signatures: list[str] = []
        for message in ctx.messages:
            if not isinstance(message, ModelResponse):
                continue
            calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if calls:
                signatures.append(
                    json.dumps(
                        [[c.tool_name, c.args] for c in calls],
                        sort_keys=True,
                        default=str,
                    )
                )

        recent = signatures[-NO_PROGRESS_LIMIT:]
        if len(recent) == NO_PROGRESS_LIMIT and len(set(recent)) == 1:
            raise NoProgress(
                f"the same tool call repeated {NO_PROGRESS_LIMIT} times with "
                f"identical arguments and no new information"
            )
        return request_context


def _register_output_cap(hooks: Hooks) -> None:
    """R3's other half: cap runaway model output as it arrives.

    S8 doubles its response every turn, and the bulk of the transcript ends up
    being the model's own prose rather than tool output. Compaction alone cannot
    fix that -- measured, it recovered 4% -- because there is nothing to digest.
    Part A capped model output at the point it was appended to the transcript;
    `after_model_request` is the framework's equivalent seam, and returning a
    modified response here is what keeps the cap in the persisted history rather
    than only in the outgoing view.
    """

    @hooks.on.after_model_request
    async def cap_output(ctx: RunContext, *, request_context, response: ModelResponse):
        parts = []
        capped = False
        for part in response.parts:
            if isinstance(part, TextPart) and len(part.content) > config.MAX_ASSISTANT_CHARS:
                capped = True
                parts.append(
                    dataclasses.replace(
                        part,
                        content=part.content[: config.MAX_ASSISTANT_CHARS]
                        + f"\n[model output truncated at "
                        f"{config.MAX_ASSISTANT_CHARS} characters; "
                        f"it was {len(part.content)}]",
                    )
                )
            else:
                parts.append(part)
        if not capped:
            return response
        ctx.deps.record(ctx.run_step, "response_truncated", {"cap": config.MAX_ASSISTANT_CHARS})
        return dataclasses.replace(response, parts=parts)


def _register_error_hooks(hooks: Hooks) -> None:
    """F4: make the framework's swallowed failures observable.

    Each hook re-raises. Their only job is to record that the failure happened
    before the framework absorbs it into a retry prompt and reports success.
    """

    @hooks.on.tool_validate_error
    async def on_validate_error(ctx: RunContext, *, call, tool_def, args, error):
        ctx.deps.tool_retries += 1
        ctx.deps.record_swallowed("tool_validate", error, detail=call.tool_name)
        raise error

    @hooks.on.tool_execute_error
    async def on_execute_error(ctx: RunContext, *, call, tool_def, args, error):
        ctx.deps.record_swallowed("tool_execute", error, detail=call.tool_name)
        raise error

    @hooks.on.model_request_error
    async def on_model_error(ctx: RunContext, *, request_context, error):
        ctx.deps.record_swallowed("model_request", error)
        raise error
