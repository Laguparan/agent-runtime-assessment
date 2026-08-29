"""R1 and R5: the agent loop, and the things that stop it.

Ordering is the load-bearing part of this file. Within a step:

    1. persist the model's response          <- BEFORE any tool runs
    2. for each call: persist the result     <- AFTER the tool runs
    3. move to the next step

A `kill -9` between 1 and 2 is the dangerous window, and it is the one R2 is
graded on. On resume the recorded response is replayed rather than re-requested,
so the same tools are re-attempted with the same (run_id, step, index) -- which
means the same idempotency key, which means `storage.commit_email` finds the
row already there and does not send again. The effect is committed in the same
transaction as its ledger entry, so there is no ordering in which the email
exists but the ledger entry does not.

A crash between the model answering and step 1 loses the response entirely and
resume asks again. That is safe because no tool has run yet; it is also the
reason the response is persisted before anything else happens.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import time
from typing import Any

from . import client, config, memory, storage, tools
from .policy import RunPolicy
from .trace import Tracer


class StopReason(str, enum.Enum):
    COMPLETED = "completed"
    STEP_LIMIT = "step_limit"
    NO_PROGRESS = "no_progress"
    CONTEXT_EXHAUSTED = "context_exhausted"
    COST_BUDGET = "cost_budget"
    TIME_BUDGET = "time_budget"
    MODEL_UNAVAILABLE = "model_unavailable"
    RUNTIME_ERROR = "runtime_error"


@dataclasses.dataclass
class RunOutcome:
    run_id: str
    stop_reason: StopReason
    detail: str
    steps: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    policy_denials: int = 0
    emails_sent: int = 0
    compactions: int = 0

    def summary(self) -> str:
        return (
            f"{self.run_id}: {self.stop_reason.value} after {self.steps} steps "
            f"({self.detail}). tokens in/out {self.input_tokens}/{self.output_tokens}, "
            f"${self.cost_usd:.4f}, {self.retries} retries, "
            f"{self.tool_calls} tool calls ({self.tool_failures} failed, "
            f"{self.policy_denials} denied), {self.emails_sent} emails."
        )

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["stop_reason"] = self.stop_reason.value
        return data


# --------------------------------------------------------------- projections


def _project(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the transcript from (responses, results). A pure function of the log.

    Message order follows step and call index, never the order rows happened to
    land in the log. Under chaos those differ: a resume fills in a tool result
    for step 2 after step 3 has already been recorded, and appending it would
    put step 2's result after step 3's. The model would see a transcript that
    never happened.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": state["task"]},
    ]

    for step in sorted(state["responses"]):
        response = state["responses"][step]
        messages.append(
            {
                "role": "assistant",
                "content": response.get("text", ""),
                "tool_calls": response.get("calls", []),
            }
        )
        for call in response.get("calls", []):
            result = state["results"].get((step, call["index"]))
            if result is None:
                continue  # not run yet; the loop is about to run it
            messages.append(
                {
                    "role": "tool",
                    "tool_use_id": result.get("call_id", ""),
                    "name": result.get("tool", ""),
                    "content": result["content"],
                    "is_error": not result.get("ok", False),
                }
            )
    return messages


def _first_incomplete(state: dict[str, Any]) -> int | None:
    """Earliest step whose recorded response has a tool call with no result.

    This is the resume point. Taking `last recorded step + 1` instead is the bug
    that silently drops a send: a process killed between recording a response
    and running its tools leaves exactly this gap, and skipping it means the
    tool never runs at all. Not twice is only half of exactly-once.
    """
    for step in sorted(state["responses"]):
        for call in state["responses"][step].get("calls", []):
            if (step, call["index"]) not in state["results"]:
                return step
    return None


def _rebuild(conn, run_id: str) -> dict[str, Any]:
    """Fold the event log into the state the loop needs to carry on.

    This is the only way run state is ever reconstructed. There is no separate
    checkpoint that could disagree with the log.
    """
    state: dict[str, Any] = {
        "task": "",
        "messages": [],
        "step": 0,
        "responses": {},       # step -> recorded model response payload
        "results": {},         # (step, index) -> recorded tool result payload
        "input_tokens": 0,
        "output_tokens": 0,
        "retries": 0,
        "compactions": 0,
        "ended": None,         # set if this run already reached a stop reason
    }

    for event in storage.iter_events(conn, run_id):
        kind, payload, step = event["kind"], event["payload"], event["step"]

        if kind == "run_started":
            state["task"] = payload["task"]
        elif kind == "model_response":
            state["responses"][step] = payload
            state["input_tokens"] += payload.get("input_tokens", 0)
            state["output_tokens"] += payload.get("output_tokens", 0)
        elif kind in ("tool_result", "policy_denied"):
            state["results"][(step, payload["index"])] = payload
        elif kind == "model_retry":
            state["retries"] += 1
        elif kind == "run_ended":
            state["ended"] = payload
        elif kind == "compaction":
            state["compactions"] += 1
            # Compaction is a pure view over the transcript, so it is recomputed
            # rather than stored. Only the count is carried forward, for the log.

    incomplete = _first_incomplete(state)
    last = max(state["responses"], default=-1)
    state["step"] = incomplete if incomplete is not None else last + 1
    state["messages"] = _project(state)
    return state


def _signatures(state: dict[str, Any]) -> list[str]:
    """Per-step tool signatures in step order, for no-progress detection."""
    return [
        state["responses"][step].get("signature", "")
        for step in sorted(state["responses"])
    ]


def _signature(calls: list[dict[str, Any]]) -> str:
    """Canonical identity of a step's tool calls, for no-progress detection.

    Excludes the model's tool_use ids on purpose. S4 loops with identical ids
    and S9 reuses ids across different calls; neither tells you whether the
    agent is making progress. What the tools were asked to do does.
    """
    return json.dumps(
        [[c.get("name"), c.get("args"), c.get("raw")] for c in calls], sort_keys=True
    )


# --------------------------------------------------------------------- loop


class AgentRun:
    def __init__(
        self,
        run_id: str,
        task: str,
        policy: RunPolicy,
        conn,
        tracer: Tracer,
        scenario: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.policy = policy
        self.conn = conn
        self.tracer = tracer
        self.scenario = scenario
        self.verbose = verbose
        self.started_at = time.time()
        self.outcome_counters = {
            "tool_calls": 0,
            "tool_failures": 0,
            "policy_denials": 0,
            "emails_sent": 0,
            "retries": 0,
        }
        self.client = client.ModelClient(
            session=run_id, scenario=scenario, on_retry=self._on_retry
        )

    # ---------------------------------------------------------------- helpers

    def _record(self, step: int, kind: str, payload: dict[str, Any]) -> None:
        """One call site for both sinks, so the log and the trace cannot drift."""
        storage.append_event(self.conn, self.run_id, step, kind, payload)
        self.tracer.emit(step, kind, payload)

    def _on_retry(self, attempt: int, reason: str, wait: float) -> None:
        self.outcome_counters["retries"] += 1
        self._record(
            self._current_step,
            "model_retry",
            {"attempt": attempt, "reason": reason, "wait_seconds": round(wait, 3)},
        )
        self._say(f"  retry {attempt}: {reason}; waiting {wait:.2f}s")

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _budget_stop(self, state: dict[str, Any]) -> tuple[StopReason, str] | None:
        elapsed = time.time() - self.started_at
        if elapsed > config.WALL_CLOCK_BUDGET:
            return StopReason.TIME_BUDGET, f"wall clock budget of {config.WALL_CLOCK_BUDGET:g}s spent"

        cost = client.estimated_cost(state["input_tokens"], state["output_tokens"])
        if cost > config.COST_BUDGET_USD:
            return StopReason.COST_BUDGET, f"simulated cost ${cost:.4f} exceeds ${config.COST_BUDGET_USD:.2f}"

        recent = _signatures(state)[-config.NO_PROGRESS_LIMIT :]
        if (
            len(recent) == config.NO_PROGRESS_LIMIT
            and len(set(recent)) == 1
            and recent[0] not in ("", "[]")
        ):
            return (
                StopReason.NO_PROGRESS,
                f"the same tool call repeated {config.NO_PROGRESS_LIMIT} times with "
                f"identical arguments and no new information",
            )
        return None

    # ------------------------------------------------------------------- run

    def execute(self, resume: bool = False) -> RunOutcome:
        self._current_step = 0

        if not resume:
            storage.create_run(
                self.conn, self.run_id, self.task, self.policy.to_dict(), self.scenario
            )
            self._record(
                0,
                "run_started",
                {
                    "task": self.task,
                    "policy": self.policy.to_dict(),
                    "scenario": self.scenario,
                    "limits": {
                        "max_steps": config.MAX_STEPS,
                        "token_ceiling": config.TOKEN_CEILING,
                        "cost_budget_usd": config.COST_BUDGET_USD,
                        "wall_clock_budget": config.WALL_CLOCK_BUDGET,
                    },
                },
            )

        state = _rebuild(self.conn, self.run_id)

        if resume and state["ended"] is not None:
            # The log says this run already reached a stop reason, so the process
            # exited on purpose rather than being killed. Continuing would drive
            # the conversation past its own ending and, worse, mint fresh
            # idempotency keys for steps that never existed.
            ended = state["ended"]
            self._say(
                f"{self.run_id} already finished: {ended.get('stop_reason')} "
                f"({ended.get('detail')}). Nothing to resume."
            )
            return RunOutcome(
                run_id=self.run_id,
                stop_reason=StopReason(ended.get("stop_reason", "runtime_error")),
                detail=str(ended.get("detail", "")),
                steps=int(ended.get("steps", 0)),
                input_tokens=int(ended.get("input_tokens", 0)),
                output_tokens=int(ended.get("output_tokens", 0)),
                cost_usd=float(ended.get("cost_usd", 0.0)),
                retries=int(ended.get("retries", 0)),
                tool_calls=int(ended.get("tool_calls", 0)),
                tool_failures=int(ended.get("tool_failures", 0)),
                policy_denials=int(ended.get("policy_denials", 0)),
                emails_sent=storage.count_emails(self.conn, self.run_id),
                compactions=int(ended.get("compactions", 0)),
            )

        if resume:
            self._say(
                f"Resuming {self.run_id} at step {state['step']} "
                f"({len(state['messages'])} messages recovered from the event log)."
            )

        try:
            reason, detail, state = self._drive(state)
        except Exception as exc:  # noqa: BLE001 - a crash still owes a stop reason
            reason, detail = StopReason.RUNTIME_ERROR, f"{exc.__class__.__name__}: {exc}"

        return self._finish(state, reason, detail)

    def _drive(self, state: dict[str, Any]) -> tuple[StopReason, str, dict[str, Any]]:
        while True:
            step = state["step"]
            self._current_step = step

            if step >= config.MAX_STEPS:
                return StopReason.STEP_LIMIT, f"reached the {config.MAX_STEPS} step ceiling", state

            stop = self._budget_stop(state)
            if stop:
                return stop[0], stop[1], state

            # --- context budget (R3) -------------------------------------
            # Compaction produces a *view* for this one request. The durable
            # transcript stays complete, so a later turn is never reasoning
            # against a history that an earlier compaction quietly destroyed --
            # and replay can recompute the same view from the same log.
            try:
                view, stats = memory.compact(state["messages"])
            except memory.ContextExhausted as exc:
                return StopReason.CONTEXT_EXHAUSTED, str(exc), state

            if stats["compacted"]:
                state["compactions"] += 1
                self._record(step, "compaction", stats)
                self._say(f"  compacted context {stats['before']} -> {stats['after']} tokens")

            # --- model turn ----------------------------------------------
            recorded = state["responses"].get(step)
            if recorded is None:
                self._say(f"\n--- step {step} ---")
                try:
                    response = self.client.complete(
                        view, tools.tool_descriptions(), step=step
                    )
                except client.ModelUnavailable as exc:
                    return StopReason.MODEL_UNAVAILABLE, str(exc), state
                except client.ModelRefused as exc:
                    return StopReason.MODEL_UNAVAILABLE, str(exc), state

                # S8: cap runaway model output on arrival. Done here, not in
                # compaction, so the cap is recorded as a fact about the turn
                # rather than something compaction quietly did later.
                text = response.text
                if len(text) > config.MAX_ASSISTANT_CHARS:
                    self._record(
                        step,
                        "response_truncated",
                        {"original_chars": len(text), "kept_chars": config.MAX_ASSISTANT_CHARS},
                    )
                    text = (
                        text[: config.MAX_ASSISTANT_CHARS]
                        + f"\n[model output truncated at {config.MAX_ASSISTANT_CHARS} "
                          f"characters; it was {len(response.text)}]"
                    )

                calls = [
                    {
                        "id": call.id,
                        "name": call.name,
                        "index": call.index,
                        "args": call.args,
                        "raw": call.raw,
                        "parse_error": call.parse_error,
                    }
                    for call in response.calls
                ]
                recorded = {
                    "text": text,
                    "calls": calls,
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "attempts": response.attempts,
                    "signature": _signature(calls),
                }
                # Persisted before a single tool runs. This is the resume anchor.
                self._record(step, "model_response", recorded)

                state["responses"][step] = recorded
                state["input_tokens"] += response.input_tokens
                state["output_tokens"] += response.output_tokens
                state["messages"] = _project(state)
            else:
                self._say(f"\n--- step {step} (replayed from the event log) ---")

            if recorded.get("text"):
                self._say(f"  model: {recorded['text'][:160]}")

            if not recorded["calls"]:
                state["step"] = step + 1
                return StopReason.COMPLETED, "the model finished without requesting tools", state

            # --- tool turn -----------------------------------------------
            self._run_calls(step, recorded["calls"], state)
            state["step"] = step + 1

            stop = self._budget_stop(state)
            if stop:
                return stop[0], stop[1], state

    def _run_calls(self, step: int, calls: list[dict[str, Any]], state: dict[str, Any]) -> None:
        for call in calls:
            index = call["index"]

            already = state["results"].get((step, index))
            if already is not None:
                # Resume landed on a call whose result is already in the log.
                self._say(f"  tool {call['name']}: result recovered from the log")
                continue

            payload = self._execute_one(step, call)
            kind = "policy_denied" if payload.get("denied") else "tool_result"
            self._record(step, kind, payload)

            state["results"][(step, index)] = payload
            # Re-project rather than append: under resume this result may belong
            # at an earlier position than events already folded in.
            state["messages"] = _project(state)

    def _execute_one(self, step: int, call: dict[str, Any]) -> dict[str, Any]:
        name, index = call["name"], call["index"]
        self.outcome_counters["tool_calls"] += 1

        # S2: the arguments never parsed. The model still made the call, so it
        # gets a result -- an error explaining exactly what was wrong with it.
        if call["args"] is None:
            self.outcome_counters["tool_failures"] += 1
            detail = call.get("parse_error") or "arguments were not valid JSON"
            raw = (call.get("raw") or "")[:200]
            self._say(f"  tool {name}: malformed arguments ({detail})")
            return {
                "index": index,
                "call_id": call.get("id", ""),
                "tool": name,
                "ok": False,
                "verdict": "malformed",
                "content": tools.envelope(
                    name,
                    False,
                    f"The arguments for {name} were not valid JSON and could not be "
                    f"used: {detail}. Received: {raw!r}. Re-send the call with a "
                    f"single well-formed JSON object.",
                ),
                "malformed": True,
            }

        ctx = tools.ToolContext(self.run_id, step, index, self.policy, self.conn)
        result = tools.execute(ctx, name, call["args"])

        if not result.ok:
            self.outcome_counters["tool_failures"] += 1
        if name == "send_email" and result.ok and not result.replayed:
            self.outcome_counters["emails_sent"] += 1

        denied = result.verdict == tools.VERDICT_DENIED
        if denied:
            self.outcome_counters["policy_denials"] += 1

        status = "ok" if result.ok else "error"
        self._say(f"  tool {name}: {status} - {result.content[:120]}")

        return {
            "index": index,
            "call_id": call.get("id", ""),
            "tool": name,
            "args": call["args"],
            "ok": result.ok,
            "verdict": result.verdict,
            "denied": denied,
            "replayed": result.replayed,
            "content": tools.envelope(name, result.ok, result.content),
        }

    def _finish(self, state: dict[str, Any], reason: StopReason, detail: str) -> RunOutcome:
        outcome = RunOutcome(
            run_id=self.run_id,
            stop_reason=reason,
            detail=detail,
            steps=state["step"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            cost_usd=client.estimated_cost(state["input_tokens"], state["output_tokens"]),
            # state["retries"] is what the log already held (non-zero on resume);
            # the counter is what this process added.
            retries=state["retries"] + self.outcome_counters["retries"],
            compactions=state["compactions"],
            emails_sent=storage.count_emails(self.conn, self.run_id),
            tool_calls=self.outcome_counters["tool_calls"],
            tool_failures=self.outcome_counters["tool_failures"],
            policy_denials=self.outcome_counters["policy_denials"],
        )
        self._record(state["step"], "run_ended", outcome.to_dict())
        self._say("\n" + outcome.summary())
        return outcome


def run_agent(
    run_id: str,
    task: str,
    policy: RunPolicy | None = None,
    scenario: str | None = None,
    resume: bool = False,
    db_path: str | None = None,
    trace_dir: str | None = None,
    verbose: bool = True,
) -> RunOutcome:
    storage.init_db(db_path)
    conn = storage.connect(db_path)
    try:
        existing = storage.get_run(conn, run_id)
        if resume:
            if existing is None:
                raise SystemExit(f"no run named {run_id} to resume")
            task = existing["task"]
            policy = RunPolicy.from_dict(json.loads(existing["policy_json"]))
            scenario = existing["scenario"]
        elif existing is not None:
            # Starting fresh on an id that already has an event log would fold
            # the old run's events into the new state and silently continue it.
            # Worse, the idempotency keys would collide, so a send from the old
            # run would suppress a genuinely new one.
            raise SystemExit(
                f"run {run_id} already exists. Use "
                f"'python -m agent.cli resume {run_id}' to continue it, or pick "
                f"a different --run-id."
            )

        with Tracer(run_id, trace_dir) as tracer:
            run = AgentRun(
                run_id, task, policy or RunPolicy(), conn, tracer, scenario, verbose
            )
            return run.execute(resume=resume)
    finally:
        conn.close()
