# Time log

> **Fill in the hours before submitting.** The phases and order below are
> accurate — they are what was actually built, in the order the commits show.
> The durations are the one thing I cannot fill in for you, and inventing them
> would be precisely the dishonesty the brief warns about. Replace every `__`.

**Cap:** 6 hours for Part A, 2 hours for Part B. **Actual:** `__` / `__`

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

## Part B — Pydantic AI (2 hour cap)

| # | Phase | Time | Notes |
|---|---|---|---|
| 11 | Probing the framework's seams before writing anything | `__` | Cheapest hour of the whole exercise. See below. |
| 12 | Wiring Pydantic AI to the local mock via its Anthropic provider | `__` | Worked first try; the awkward part was per-run headers, not the endpoint. |
| 13 | F2 — durability around a loop I don't own | `__` | Most of the Part B time. |
| 14 | F3 — trust boundary moved into hooks | `__` | Fast; the policy code was reused unchanged. |
| 15 | F4 — instrumenting what the framework hides | `__` | |
| 16 | F1 — compaction and the no-progress hook | `__` | |
| 17 | `FRAMEWORK.md` | `__` | |

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

## Part B: where the time actually went

**Probing before building was the highest-leverage thing I did.** Before writing a
line of `agent_fw/`, I checked three things against the installed version: whether
there was an interception point around tool execution (there is,
`wrap_tool_execute`), what `RunContext` exposes, and whether the Anthropic provider
would talk to the mock at all. The third took one throwaway script and worked. The
second saved the build: `run_step` looks like the obvious idempotency anchor and
restarts at 1 on resume, which I found in a ten-line spike rather than in a chaos
run at hour two.

**F2 took over half the Part B budget**, which is the honest headline. Everything
else was reuse or configuration. Three separate things had to be discovered by
measurement rather than by reading: the step counter resets, history is only
persisted on the success path unless you checkpoint yourself, and the framework
offers no hook to attach a per-request header — so the mock's session addressing
had to go on the httpx transport.

**What Part A bought me here.** The policy, path confinement, the effect ledger,
every tool body and the tracer were imported unchanged. Part B is ~600 lines of
glue rather than a rewrite, and that is the entire answer to F5's exit-cost
question.

## Deferred, with reasons

- **Streaming responses.** The mock does not stream, so the client does not
  either. A real Messages API stream would change what "reset at a random byte
  offset" looks like on the client side.
- **`run_python` isolation beyond a wall clock.** Needs a container; out of
  scope for a laptop exercise. Named as unsafe in `DECISIONS.md` rather than
  half-built.
- **Per-path locking.** Demonstrated as a failing eval (F01) instead of fixed.
- **Part B evals.** The brief does not ask for them, and the chaos harness already
  runs against both builds (`--module agent_fw.cli`). `make eval` still exercises
  Part A only.
- **Part B replay.** `agent replay` has no equivalent in the framework build. The
  framework owns the decision points, and reproducing them offline would mean
  reimplementing its control flow — which is most of what the framework is for.
