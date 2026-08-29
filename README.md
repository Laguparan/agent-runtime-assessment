# Adversarial Agent Runtime — Parts A and B

A durable, sandboxed, replayable agent runtime built without a framework.
Python 3.11+, standard library plus PyYAML (used only by the mock server).

## First, the thing you need to know

`mockllm/`, `harness/chaos.py` and `harness/redteam/` are listed in the brief
under **"What you're given"**. They never arrived. I searched the machine — no
`mockllm/`, no `harness/`, no `scenarios/`.

So `mockllm_local/` is mine: a mock model server built from the S1–S12 table in
the brief and nothing else. It is named differently on purpose so it cannot be
mistaken for the provided package or read as a modification of it. The chaos
harness in `harness/chaos.py` is mine too.

**Everything below was measured against my reconstruction.** Where that weakens
a claim, it is said so at the point of the claim.

## Run it

```bash
make setup     # pip install pyyaml, initialise the database
make test      # 44 offline unit tests, no sockets — passes with networking off
make eval      # 21 eval cases; starts its own mock server on :8765
make chaos     # 100 kill/resume cycles; starts its own mock server on :8766
```

**No `make` on Windows?** Every target is a single command, so run them directly:

```bash
python -m pip install pyyaml && python -m agent.storage   # setup
python -m unittest discover -s tests -v                   # test
python evals/eval_suite.py                                # eval
python harness/chaos.py -n 100                            # chaos
python -m mockllm_local                                   # serve
python -m mockllm_local.smoke_test                        # smoke
```

`make eval` and `make chaos` are self-contained. Only interactive runs need a
server of your own:

```bash
make serve                       # terminal 1 — :8000, default scenario S1
make smoke                       # terminal 2 — 19 checks on the mock itself

python -m agent.cli run --task "summarise notes.txt" --scenario S7
python -m agent.cli run --task "email the report" --allow-email team@example.com
python -m agent.cli resume <run_id>
python -m agent.cli replay <run_id>      # no server needed
python -m agent.cli inspect <run_id>
python -m agent.cli emails
```

**`send_email` is refused unless `--allow-email` names the recipient.** That is
the R4 trust boundary, not a convenience flag — see `agent/policy.py`.

## Results, as measured

| | |
|---|---|
| `make test` | 44 passed |
| `make eval` | 19 passed, 2 failed **on purpose** (F01, F02), 0 regressions vs baseline |
| `make chaos` | 100 iterations, 87–94 killed mid-run, **exactly 2 emails every time** |
| `make smoke` | 19/19 — the mock misbehaves as advertised |
| S1–S12 | all survived without a crash or a corrupted transcript |

The two failing evals are deliberate. A fully green board would mean the suite
is too easy. They are explained in `DECISIONS.md`.

## How it is put together

```
agent/
  config.py    limits and budgets, in one place
  policy.py    R4 — the capability set, frozen at t=0, never re-read from content
  paths.py     workspace confinement via realpath + commonpath
  tools.py     five tools, argument validation, sandboxing, idempotency keys
  storage.py   R2 — append-only event log and the exactly-once effect ledger
  trace.py     R6 — fsynced JSONL, one line per event
  tokens.py    tokenizer resolution: mockllm → mockllm_local → estimate
  memory.py    R3 — four-pass compaction, by kind first and position second
  client.py    resilient transport; classifies S5/S6/S12 apart from a real answer
  loop.py      R1/R5 — the loop, its budgets, and its stop reasons
  replay.py    R6 — re-derives decisions from a trace with no server
  cli.py       run / resume / replay / inspect / emails / runs
evals/         21 cases + stored baseline
tests/         44 offline unit tests
harness/       chaos.py — the kill/resume/assert loop
mockllm_local/ the mock server (see its own README)
```

Three ideas carry most of the weight:

**The event log is the only state.** Everything the loop needs — transcript,
step, which tools already ran — is folded out of an append-only table. There is
no checkpoint that could disagree with it.

**The effect and its ledger entry commit together.** `send_email` writes to
`emails` and `effects` in one transaction, so there is no ordering in which the
email exists but the ledger does not know. `synchronous=FULL` makes that true
across a power cut, not just a `kill -9`.

**Privileges are frozen before untrusted bytes arrive.** In S7 the model
complies fully with the injection. All three vectors are refused, and the
refusal never depends on recognising the injection as one.

## What does not work

- **`run_python` is not a real sandbox.** The network block is a monkeypatch in
  the child's namespace and is defeatable. The memory cap uses `setrlimit`,
  which is POSIX-only — on Windows it is **not enforced at all**. Only the 5s
  wall clock is real.
- **A granted capability is unbounded within its grant.** Recipients are gated;
  payload contents are not. An injection can still choose *what* gets emailed to
  an authorised address.
- **No file locking** — concurrent writes to one workspace path interleave.
  Eval F01 demonstrates it rather than hiding it.
- **A model claiming a failed tool succeeded is not flagged** during the run
  (S11). The trace records the ground truth, so it is auditable afterwards.
  Eval F02.
- **No streaming.** The mock does not stream, so neither does the client.
- **Token counts are not the real ones.** `mockllm_local/tokenizer.py` will not
  agree digit-for-digit with the assessment's. The budget is set to compact at
  6,000 against an 8,000 ceiling to leave headroom.
- **Part B compaction cannot claim the 8k ceiling.** `agent_fw` measures token
  pressure with our tokenizer while the framework enforces its limit with its
  own count. The two numbers do not correspond, so compaction there reduces
  pressure without being able to prove it stays under the ceiling. The framework's
  `per_request_input_tokens_limit` is what actually enforces it.
- **No replay for Part B.** `agent replay` has no framework equivalent; the
  framework owns the decision points. Deferred with a reason in `TIMELOG.md`.

## Part B — the same runtime on Pydantic AI

`agent_fw/` rebuilds the runtime on **Pydantic AI 2.35**, pointed at the same mock
server through its Anthropic provider. It reuses everything under `agent/` that is
not loop control — the policy, path confinement, the effect ledger, every tool body
and the tracer are imported unchanged — so the framework owns orchestration and
nothing else.

```bash
make chaos-fw                # the same 100 kill/resume cycles, driving agent_fw
make run-fw SCENARIO=S7      # interactive; needs `make serve`

python -m agent_fw.cli run --task "email the report" --allow-email team@example.com
python -m agent_fw.cli resume <run_id>
```

Measured, same harness as Part A:

| | |
|---|---|
| S1–S12 | all survived |
| `make chaos-fw` | 30 iterations, all 30 killed mid-run, **exactly 2 emails every time** |
| S7 injection | 3 vectors denied, 0 emails, with email granted to another recipient |

Three findings that shaped the build, all in `FRAMEWORK.md` with numbers:
`RunContext.run_step` restarts at 1 on resume and cannot anchor an idempotency key;
`usage.requests` reported 2 while the wire saw 4; and there is no hook to add a
per-request header, so the mock's session addressing had to go on the transport.

`DECISIONS.md` names three ways Part A is still unsafe and defends the compaction
strategy against the obvious alternative. `FRAMEWORK.md` is the Part B write-up.
`TIMELOG.md` still needs its hours filled in.
