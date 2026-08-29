# Time log

> **Fill in the hours before submitting.** The phases and order below are
> accurate — they are what was actually built, in the order the commits show.
> The durations are the one thing I cannot fill in for you, and inventing them
> would be precisely the dishonesty the brief warns about. Replace every `__`.

**Cap:** 6 hours for Part A. **Actual:** `__`

| # | Phase | Time | Notes |
|---|---|---|---|
| 0 | Reading the brief, discovering `mockllm/` was missing | `__` | Searched the machine; no `mockllm/`, `harness/`, or `scenarios/` anywhere. |
| 1 | Building `mockllm_local/` — server, scenarios S1–S12, tokenizer | `__` | Not in the original plan. Unavoidable: nothing else can be tested without it. |
| 2 | Durability layer — schema, event log, effect ledger | `__` | Written before the loop, so the exactly-once design drove the loop rather than the reverse. |
| 3 | Policy, path confinement, tool sandbox | `__` | |
| 4 | Compaction (R3) | `__` | Rewritten once after the turn-3 eval failed. |
| 5 | Resilient client (S5/S6/S12) and the agent loop | `__` | |
| 6 | Chaos harness and the resume bug it found | `__` | See below. |
| 7 | Replay (R6) | `__` | |
| 8 | Eval suite (R7) — 21 cases | `__` | |
| 9 | Offline unit tests — 44 cases | `__` | |
| 10 | Write-ups | `__` | |

## Where the time actually went

Two things cost more than expected, and both were worth it.

**Rebuilding the mock server.** Roughly a third of the work, none of it in the
brief's requirements. The alternative was writing a runtime against a server
whose behaviour I could only guess at, and never once running it.

**The chaos harness paid for itself immediately.** Writing it before trusting
the resume logic — which the brief explicitly says scores well and costs
nothing — surfaced a real bug on the first hundred-iteration run: 6 runs in 100
silently dropped a `send_email`, because resume restarted at `last recorded
step + 1` rather than at the earliest step with an unanswered tool call. That is
the failure mode R2 is graded on, it only appears under a kill in a window of a
few milliseconds, and no amount of reading the code would have found it.

## Deferred, with reasons

- **Streaming responses.** The mock does not stream, so the client does not
  either. A real Messages API stream would change what "reset at a random byte
  offset" looks like on the client side.
- **`run_python` isolation beyond a wall clock.** Needs a container; out of
  scope for a laptop exercise. Named as unsafe in `DECISIONS.md` rather than
  half-built.
- **Per-path locking.** Demonstrated as a failing eval (F01) instead of fixed.
- **Part B.** Not started; `agent/framework_agent.py` and `agent/FRAMEWORK.md`
  are from an earlier attempt and predate this runtime. They do not run against
  it and should not be read as a Part B submission.
