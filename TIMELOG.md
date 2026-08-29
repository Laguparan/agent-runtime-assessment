# Time log

**Part A: 4h 22m** (cap 6h) · **Part B: ~1h** (cap 2h, on the `Architecture` branch)

## How these numbers were produced

They are **reconstructed from git reflog entries and file timestamps**, not
stopwatched while working. The anchors are real and checkable:

| Anchor | Timestamp |
|---|---|
| First scaffolding commit | 08-28 22:53 |
| First hand-written Part A commit | 08-28 23:48 |
| First Part B sketch commit | 08-29 01:57 |
| Brief re-read, rebuild begins | 08-29 12:31 |
| Mock server runs for the first time | 08-29 13:21 |
| Part A rebuild committed and pushed | 08-29 13:49 |
| First Part B framework spike | 08-29 21:30 |
| Part B committed and pushed | 08-29 22:08 |

Block totals between those anchors are solid. **Per-phase rows inside a block
are apportioned, not separately measured** — the work overlapped rather than
running strictly in sequence, so read them as a breakdown of a measured total
rather than as seven independent stopwatch readings.

**AI assistance was used heavily throughout**, which the brief explicitly
permits and which is the main reason the elapsed times are short. The figures
below are wall-clock on the project, not an estimate of how long this would
take by hand.

---

## Part A — first attempt, hand-written (3h 04m)

08-28 22:53 → 08-29 01:57. Scaffolding, a procedural loop, a tool sandbox, a
SQLite schema, and a stubbed eval suite. This is the version that got replaced;
it is still in the history at `962014b` and `d2cc3a5`.

## Part A — rebuild (1h 18m)

**Block 1 — 12:31 → 13:21 (50m): nothing could be tested yet**

| # | Phase | Time | Notes |
|---|---|---|---|
| 0 | Reading the brief, discovering `mockllm/` was missing | 10m | Searched the machine: no `mockllm/`, `harness/`, or `scenarios/` anywhere. |
| 1 | Building `mockllm_local/` — server, S1–S12, tokenizer | 40m | Not in the brief's requirements. Unavoidable: the first block ends the moment a server finally answers. |

**Block 2 — 13:21 → 13:49 (28m): everything that needed a server**

| # | Phase | Time | Notes |
|---|---|---|---|
| 2 | Durability — schema, event log, effect ledger | 5m | Written before the loop, so exactly-once drove the loop rather than the reverse. |
| 3 | Policy, path confinement, tool sandbox | 3m | |
| 4 | Compaction (R3) | 3m | Rewritten once after the turn-3 eval failed. |
| 5 | Resilient client (S5/S6/S12) and the agent loop | 5m | |
| 6 | Chaos harness, and the dropped send it found | 4m | See below. |
| 7 | Replay (R6) | 2m | |
| 8 | Eval suite (R7) — 21 cases | 3m | |
| 9 | Offline unit tests — 44 cases | 2m | |
| 10 | Write-ups, Makefile, commits | 1m | |

## Part B — Pydantic AI (~1h)

**21:30 → 22:08 (38m) build, plus ~20m verification later the same evening.**

Part B is not on this branch. It lives on **`Architecture`**, which is this
branch plus `agent_fw/` and `FRAMEWORK.md`; the per-phase breakdown is in the
copy of this file there.

---

## Where the time actually went

**Rebuilding the mock server was the single largest line item** — 40 of the
first 50 minutes, and none of it in the brief's requirements. The alternative
was writing a runtime against a server whose behaviour I could only guess at
and never once running it.

**The chaos harness paid for itself immediately.** Writing it before trusting
the resume logic — which the brief says scores well and costs nothing — surfaced
a real bug on the first hundred-iteration run: 6 runs in 100 silently dropped a
`send_email`, because resume restarted at `last recorded step + 1` rather than
at the earliest step with an unanswered tool call. That is the exact failure R2
is graded on, it appears only in a window of a few milliseconds, and no amount
of reading the code would have found it.

**Part B: probing before building was the highest-leverage decision.** Before
writing a line of `agent_fw/`, I checked whether there was an interception point
around tool execution (there is), what `RunContext` exposes, and whether the
Anthropic provider would talk to the mock at all. The second saved the build:
`RunContext.run_step` looks like the obvious idempotency anchor and restarts at
1 on resume. A ten-line spike found that, instead of a chaos run at minute 30.

**F2 took over a third of Part B.** Everything else was reuse or configuration.
Three things had to be found by measurement rather than by reading the docs: the
step counter resets, history is only persisted on the success path, and there is
no hook to attach a per-request header — so the mock's session addressing had to
go on the httpx transport.

**What Part A bought Part B.** The policy, path confinement, the effect ledger,
every tool body and the tracer were imported unchanged. Part B is ~600 lines of
glue rather than a rewrite, which is the whole of F5's exit-cost answer.

## Deferred, with reasons

- **Streaming responses.** The mock does not stream, so the client does not
  either. A real Messages API stream would change what "reset at a random byte
  offset" looks like on the client side.
- **`run_python` isolation beyond a wall clock.** Needs a container; out of scope
  for a laptop exercise. Named as unsafe in `DECISIONS.md` rather than half-built.
- **Per-path locking.** Demonstrated as a failing eval (F01) instead of fixed.
- **Part B evals.** The brief does not ask for them, and the chaos harness runs
  against both builds (`--module agent_fw.cli`). But nothing in `tests/` or
  `evals/` imports `agent_fw`, so a regression there would not be caught.
- **Part B replay.** `agent replay` has no framework equivalent. The framework
  owns the decision points, and reproducing them offline would mean
  reimplementing its control flow — which is most of what the framework is for.
