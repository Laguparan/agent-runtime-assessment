# Part B — Pydantic AI

**Framework:** Pydantic AI 2.35, pointed at the same `mockllm_local` server via its
Anthropic provider (`/v1/messages`). **Why:** I needed one interception point above
all — *around tool execution* — because that is where exactly-once and the trust
boundary must live once you no longer own the loop. It has one
(`wrap_tool_execute`). All numbers below are measured against S1–S12.

## Three things it did better

**1. Malformed-call recovery is free.** S2 sends three unparseable argument blobs;
S3 a nonexistent tool and three wrong-typed calls. The framework validates against
the tool's type hints and feeds a retry prompt back itself. Both completed. Part A
needed ~60 lines of `validate()` plus a loop branch; here it was `retries=6`.

**2. Transport resilience is free.** S5 (two mid-body connection resets) and S6
(429 with Retry-After, then 529) both passed with no retry code of mine. Part A's
`client.py` is ~90 lines of failure classification, backoff, and Retry-After
clamping that the SDK does underneath.

**3. Budgets are declarative primitives.** `UsageLimits(request_limit=...,
per_request_input_tokens_limit=8000)` enforces R5's step ceiling and R3's token
ceiling as configuration, where Part A hand-checks both in a `_budget_stop()` I had
to remember to call. Measured: S8 stopped with `per_request_input_tokens_limit of
8000 (request_input_tokens=8520)` — the ceiling enforced itself.

## Three places the abstraction leaked

**1. `RunContext.run_step` restarts at 1 on resume.** It counts model requests in
*this process*, not in the logical run. Measured, resuming a conversation that had
already made four calls:

```
fresh    read_file(1) read_file(2) send_email(3) send_email(4)
resumed  ...          ...          send_email(1) send_email(2)
```

Keying idempotency on it — the natural port of Part A's `(run_id, step, index)` —
gives the resumed send the key of the *first*, suppressing the second as a
duplicate: exactly-once failing in the "never" direction. **Cost:** `identity.py`,
deriving position from `ToolReturnPart` count in the persisted conversation. The
subtlest code in the build.

**2. Usage accounting stops at the framework boundary.** `usage.requests` reported
**2** while the wire saw **4**, for both S5 and S6 — half the traffic invisible,
because the SDK retries below the framework. And on any failure path there is no
`result` object at all, so usage vanishes: S4 reported 0 steps and 0 tokens until I
threaded my own `RunUsage` in. **Cost:** an httpx event-hook layer (`telemetry.py`)
to count real requests, and a `usage=` object passed into every run. F4 asks for
"every retry the framework performed on your behalf" — it will not tell you, and
you have to instrument the layer beneath it.

**3. Checkpoint granularity is a whole turn coarser.** Part A persisted each tool
result the instant it happened. There is no seam between "tool returned" and "turn
assembled" that yields a persistable message, so the tightest checkpoint is
`before_model_request`. **Cost:** a crash between a tool committing and the next
model request loses that return, so the resumed run re-issues the call — and the
*only* thing preventing a duplicate is the ledger check. In Part A that was defence
in depth; here it is the sole defence.

## One thing it makes unreasonably expensive

**Refusing a tool call without spending the model's retry budget.** Every refusal
path goes through `ModelRetry`, which consumes the same allowance that S2's
malformed arguments need. In Part A a policy denial was just a tool result and cost
nothing. Here, S7's three denials plus S2's three malformed calls compete for one
counter — I had to set `retries=6` to keep both passing, which weakens the retry
ceiling as a control. There is no "reject this call, tell the model, don't count
it" primitive.

## Exit cost

**Low, deliberately.** Every tool body, the policy, path confinement, the effect
ledger, the tracer and the tokenizer live under `agent/` and are *imported
unchanged*. `agent_fw/` is ~600 lines of pure orchestration glue. No framework type
crosses into `agent/`; the adapter is one function, `_ctx_for()`. Dropping Pydantic
AI means deleting `agent_fw/` and running `agent/cli.py` against the same database
— which already works today.

## Recommendation

**For the system in Part A: don't use the framework — but steal `UsageLimits`.**

The heaviest graded item is exactly-once under chaos, and every hard part of it
here came from *not owning the loop*: a process-scoped step counter, turn-coarse
checkpoints, retries I could not see. Part A's guarantee is easier to argue because
the ordering that makes it true is visible in one file.

The honest other half: its validation and transport layers are better than mine and
cost nothing, and its budget primitives beat my hand-rolled checks. If a duplicate
email merely annoyed someone, I would take the framework and the 4x speed-up without
hesitating. Irreversibility decides it, not ergonomics.

**Both builds pass the same chaos harness** (`--module agent_fw.cli`), exactly 2
emails every time: Part A over 100 iterations, Part B over 30.
