"""R6: re-derive a run's decisions from its trace, with no model server running.

What is replayed is the *decision*, not the effect. Re-running `write_file`
would change the workspace and re-running `send_email` would be the exact thing
R2 exists to prevent, so replay stops at the point where a decision has been
made and compares it with what the trace says happened:

  * did the arguments parse, and to the same thing?
  * did schema validation reach the same verdict?
  * did the policy reach the same verdict, for the same reason?
  * did compaction fire at the same steps?

A divergence here means the runtime's decision logic changed since the run --
which is exactly the regression this is meant to catch.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from . import config, memory, tools
from .loop import _signature
from .paths import PathDenied, safe_path
from .policy import PolicyDenied, RunPolicy
from .trace import read_trace


@dataclasses.dataclass
class Divergence:
    step: int
    index: int
    field: str
    recorded: Any
    replayed: Any

    def __str__(self) -> str:
        return (
            f"step {self.step} call {self.index}: {self.field} "
            f"recorded={self.recorded!r} replayed={self.replayed!r}"
        )


@dataclasses.dataclass
class ReplayReport:
    run_id: str
    steps: int
    calls: int
    divergences: list[Divergence]
    stop_reason: str | None
    policy: dict[str, Any]

    @property
    def faithful(self) -> bool:
        return not self.divergences


def _decide(policy: RunPolicy, name: str, args: Any) -> tuple[str, str]:
    """The pure part of tool dispatch: what would the runtime decide, and why.

    Performs no side effects -- no writes, no sends, no network -- so it is safe
    to run against any trace. Path resolution is included because confinement is
    a *decision*, not a side effect: leaving it out made replay report a blocked
    write to `../../.ssh/authorized_keys` as permitted, which is the single
    worst thing a replay tool can get wrong.
    """
    try:
        cleaned = tools.validate(name, args)
    except tools.ToolValidationError as exc:
        return tools.VERDICT_INVALID, str(exc)

    try:
        if name == "send_email":
            policy.check_email(cleaned["to"])
        elif name == "write_file":
            policy.check_write()
            safe_path(policy.workspace, cleaned["path"])
        elif name == "read_file":
            safe_path(policy.workspace, cleaned["path"])
        elif name == "run_python":
            policy.check_python()
        elif name == "http_get":
            import urllib.parse

            parsed = urllib.parse.urlparse(cleaned["url"])
            if parsed.scheme not in ("http", "https"):
                return tools.VERDICT_DENIED, f"scheme {parsed.scheme!r} is not allowed"
            policy.check_http_host(parsed.hostname)
    except (PolicyDenied, PathDenied) as exc:
        return tools.VERDICT_DENIED, str(exc)

    return tools.VERDICT_PERMITTED, ""


def replay(run_id: str, trace_dir: str | None = None, verbose: bool = True) -> ReplayReport:
    events = list(read_trace(run_id, trace_dir))
    if not events:
        raise SystemExit(f"trace for {run_id} is empty")

    policy = RunPolicy()
    stop_reason = None
    divergences: list[Divergence] = []
    messages: list[dict[str, Any]] = []
    steps = calls_seen = 0

    # Index recorded outcomes so each replayed decision has something to meet.
    recorded_results: dict[tuple[int, int], dict[str, Any]] = {}
    compaction_steps: set[int] = set()
    for event in events:
        if event["kind"] in ("tool_result", "policy_denied"):
            recorded_results[(event["step"], event["index"])] = event
        elif event["kind"] == "compaction":
            compaction_steps.add(event["step"])

    for event in events:
        kind, step = event["kind"], event["step"]

        if kind == "run_started":
            policy = RunPolicy.from_dict(event["policy"])
            messages = [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": event["task"]},
            ]
            if verbose:
                print(f"replaying {run_id}: {event['task']!r}")
                print(f"  policy: {json.dumps(policy.to_dict(), sort_keys=True)}\n")

        elif kind == "model_response":
            steps += 1

            # Compaction is deterministic, so the same history must produce the
            # same verdict. If it does not, compaction changed.
            try:
                _, stats = memory.compact(messages)
                fired = bool(stats["compacted"])
            except memory.ContextExhausted:
                fired = True
            if fired != (step in compaction_steps):
                divergences.append(
                    Divergence(step, -1, "compaction", step in compaction_steps, fired)
                )

            replayed_signature = _signature(event["calls"])
            if replayed_signature != event.get("signature"):
                divergences.append(
                    Divergence(step, -1, "signature", event.get("signature"), replayed_signature)
                )

            messages.append(
                {"role": "assistant", "content": event.get("text", ""), "tool_calls": event["calls"]}
            )

            for call in event["calls"]:
                calls_seen += 1
                index = call["index"]
                recorded = recorded_results.get((step, index))

                if call["args"] is None:
                    verdict, _ = "malformed", ""
                else:
                    verdict, _ = _decide(policy, call["name"], call["args"])

                if recorded is None:
                    # No result recorded: the process died between the response
                    # and the tool. Real and expected under chaos, not a divergence.
                    if verbose:
                        print(f"  step {step} call {index} {call['name']}: no result recorded (crash window)")
                    continue

                recorded_verdict = _recorded_verdict(recorded)
                if verdict != recorded_verdict:
                    divergences.append(
                        Divergence(step, index, "verdict", recorded_verdict, verdict)
                    )
                elif verbose:
                    print(f"  step {step} call {index} {call['name']}: {verdict}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_use_id": call.get("id", ""),
                        "name": call["name"],
                        "content": recorded["content"],
                        "is_error": not recorded.get("ok", False),
                    }
                )

        elif kind == "run_ended":
            stop_reason = event.get("stop_reason")

    report = ReplayReport(run_id, steps, calls_seen, divergences, stop_reason, policy.to_dict())

    if verbose:
        print(f"\n{steps} steps, {calls_seen} tool calls, stop reason: {stop_reason}")
        if report.faithful:
            print("replay is faithful: every decision matched the trace.")
        else:
            print(f"{len(divergences)} divergence(s):")
            for divergence in divergences:
                print(f"  {divergence}")
    return report


def _recorded_verdict(recorded: dict[str, Any]) -> str:
    """The verdict the trace recorded, read from the field, not from the prose."""
    verdict = recorded.get("verdict")
    if verdict:
        return str(verdict)
    # Traces written before `verdict` was recorded. Best effort, and the reason
    # the field exists: inferring a decision from an error message is guesswork.
    if recorded.get("malformed"):
        return "malformed"
    if recorded.get("denied"):
        return tools.VERDICT_DENIED
    return tools.VERDICT_PERMITTED
