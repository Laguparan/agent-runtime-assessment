"""Part B: the same runtime rebuilt on Pydantic AI.

Framework: **Pydantic AI 2.35**, pointed at the same local mock server as Part A
through its Anthropic provider (`/v1/messages`, the shape the brief specifies).

Reason for the choice: it is the framework whose seams I could actually inspect
before committing to it, and its capability/hook system is the only one of the
shortlist that offered a documented interception point *around tool execution* --
which is precisely where F2's exactly-once guarantee and F3's trust boundary have
to live when you no longer own the loop. Part A's earlier sketch also used it,
so the comparison is like-for-like.

Module map, one file per requirement so the claims in FRAMEWORK.md are traceable:

    deps.py        what every hook needs: run id, policy, connection, counters
    identity.py    F2 -- deriving a stable idempotency key without owning the loop
    durability.py  F2 -- ledger-backed tool execution and message-history persistence
    boundary.py    F3 -- the capability check, moved into a before_tool_execute hook
    telemetry.py   F4 -- what the framework reports, and what it hides
    compaction.py  R3 -- the 8k ceiling via a history processor
    runtime.py     assembles the agent, its tools, and its usage limits
    cli.py         run / resume / inspect, matching Part A's interface

Everything under `agent/` that is not loop control is reused verbatim -- storage,
policy, paths, the tool bodies, the tracer. That reuse is deliberate and is the
answer to F5's exit-cost question: the framework owns orchestration and nothing
else, so dropping it means rewriting this package and touching nothing under
`agent/`.
"""
