"""Per-run state handed to every hook and tool through `RunContext.deps`.

Pydantic AI threads a single `deps` object through the whole run, so this is the
only channel by which the runtime's own state -- the policy, the database
handle, the counters -- reaches code the framework calls on our behalf.

That single channel is also a constraint worth naming: anything a hook needs has
to be decided before `agent.run()` is called, because there is no way to reach
into a run in progress and add to it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agent.policy import RunPolicy
from agent.trace import Tracer


@dataclasses.dataclass
class FwDeps:
    run_id: str
    policy: RunPolicy
    conn: Any
    tracer: Tracer

    # F4 counters. Populated by telemetry.py and reported at the end of a run.
    http_requests: int = 0          # real requests, counted at the transport
    transport_retries: int = 0      # what the SDK retried underneath the framework
    tool_retries: int = 0           # ModelRetry the framework fed back to the model
    swallowed_errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # Mock-server addressing. The framework offers no per-request header hook,
    # so these are set on the run context and read back down at the transport.
    mock_step: int = 0
    mock_attempt: int = 0

    denials: int = 0
    replayed_effects: int = 0
    compactions: int = 0

    def record(self, step: int, kind: str, payload: dict[str, Any]) -> None:
        """Write one event to the SQLite log and the JSONL trace together.

        Part A had the same single call site for the same reason: two sinks
        written from two places drift, and the chaos harness reads the SQLite
        one while a human reads the trace.
        """
        from agent import storage

        storage.append_event(self.conn, self.run_id, step, kind, payload)
        self.tracer.emit(step, kind, payload)

    def record_swallowed(self, where: str, error: BaseException, detail: str = "") -> None:
        """An error the framework handled without letting it reach the caller."""
        self.swallowed_errors.append(
            {
                "where": where,
                "type": error.__class__.__name__,
                "message": str(error)[:400],
                "detail": detail,
            }
        )
