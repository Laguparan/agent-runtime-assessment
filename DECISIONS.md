# Decisions

## The mock server was not provided

`mockllm/`, `harness/chaos.py` and `harness/redteam/` are listed under "What
you're given" and did not arrive. I built `mockllm_local/` from the S1–S12 table
and wrote my own chaos harness. It is deliberately named differently so it
cannot be read as a modified `mockllm/`. Everything below was measured against
my reconstruction, so treat the numbers as evidence, not proof — particularly
S7, where the real redteam payloads are undisclosed and mine are invented.

## Exactly-once side effects (R2)

`send_email`'s observable effect is a row in `emails`; the ledger entry claiming
it is a row in `effects` keyed on the idempotency key. **Both are written in one
transaction**, so SQLite decides the outcome: either the send is recorded and
ledgered, or neither row exists. There is no window where the effect happened
but the ledger does not know. `synchronous=FULL` costs an fsync per commit and
buys the guarantee across a power cut as well as a `kill -9`.

*Rejected:* a two-phase `PENDING → act → COMPLETED` outbox. It is the right
design when the effect is a real SMTP call that cannot join the transaction, and
it needs startup reconciliation plus the provider's own idempotency key. Here
the effect *is* a database write, so the transaction subsumes all of that. If
`send_email` ever becomes real, this must change, and the outbox is where it goes.

The idempotency key is `sha256(run_id, step, index, tool, args)` —
deliberately **not** the model's `tool_use` id. S9 reuses one id across four
calls including two genuinely different sends; keying on it would suppress the
second and fail exactly-once in the *zero* direction.

Ordering within a step is the other half. The model's response is persisted
before any tool runs, so a resume replays the recorded response instead of
asking again, and the same logical send lands on the same key. An early version
resumed at `last recorded step + 1`, which silently dropped a send when the kill
landed between recording a response and running its tools — 6 runs in 100. The
resume point is now the earliest step with an unanswered call.

Measured: 100 iterations, 87–94 killed mid-run, exactly 2 emails every time.

## Injection resistance (R4)

Every privilege is computed at t=0 from the operator's arguments, recorded in
the event log, and never re-read from anything the model says or a tool returns.
`send_email` is refused unless `--allow-email` named the recipient before the
run started. In S7 the model does exactly what the injection asks; all three
vectors are refused, and the refusal never depends on recognising the injection.

*Rejected:* scanning tool output for injection patterns. It loses to the first
paraphrase, and worse, it makes the guard a function of the attacker's text.
The envelope in `tools.envelope()` marks provenance, but it is framing for the
model, not the control — the control is that the capability set is frozen.

Path confinement uses `realpath` + `commonpath`, not `startswith`, which admits
`workspace_evil`; and `http_get` checks `urlparse().hostname` on every redirect
hop, not a string split, which `https://api.github.com@evil.example/` defeats.

## Compaction (R3)

Four escalating passes: digest tool results outside the recent window; trim
assistant prose outside the anchor; collapse non-anchor tool output to a marker;
finally trim the recent window. Ordered **by kind first, position second** —
tool output is compressible because the model has already reacted to it and the
reaction survives in the assistant message; reasoning is where a fact stated
once lives.

That ordering is not cosmetic. My first version anchored the first N messages
and digested everything after, and it **failed** the turn-3-at-turn-40 task:
turn 3 is around message eight once each turn adds a response and a result, so
the fact fell one slot outside the anchor. Digesting all tool output first
removes far more tokens and removes them from the part nobody needs verbatim.

*Defended against recursive LLM summarisation of the middle:* it costs a model
call per compaction against a server that returns 429s and resets connections
mid-response, and it is nondeterministic — which breaks R6's replay guarantee
outright. Deterministic compaction means `agent replay` can recompute the exact
view the model saw from the log alone. Compaction is a *view*, computed per
request; the durable transcript is never mutated, so no later turn reasons
against history an earlier compaction destroyed.

## Three places this is still unsafe

1. **`run_python` is not a sandbox.** The network block is a monkeypatch in the
   child's namespace and can be defeated by re-importing at the C level. The
   memory cap uses `setrlimit`, which is POSIX-only, so on Windows — where this
   was developed — it is **not enforced at all**. Only the 5s wall clock is real.
   A real deployment needs a container or seccomp.
2. **A granted capability is unbounded within its grant.** Authorise
   `send_email` to `team@example.com` and an injection can still choose the
   *contents* — including exfiltrating the workspace to an allowed recipient.
   Recipients are gated; payloads are not.
3. **No file locking.** Concurrent writes to one workspace path interleave;
   `evals` F01 demonstrates 60 bad read-after-writes. Single-run use is fine,
   parallel tool calls on a shared path are not.

Also, and less severe: the runtime does not detect a model claiming a failed
tool succeeded (S11, F02). The trace records ground truth, so it is *auditable*
after the fact, but nothing flags it during the run.

## With two more weeks

Real process isolation for `run_python`. A content-level egress policy so a
granted `send_email` cannot carry arbitrary workspace bytes. Per-path advisory
locking. The outbox pattern behind a `send_email` that actually leaves the
machine. And a much larger redteam corpus — seven adversarial evals that I wrote
myself is a measure of my imagination, not of the runtime.

## Evals

19 pass, 2 fail on purpose (F01, F02 above), diffed against a stored baseline.
`make test` runs 44 offline unit tests with no sockets.
