# `mockllm_local` — local stand-in for the assessment's mock server

## Why this exists

The candidate brief lists `mockllm/`, `harness/chaos.py` and `harness/redteam/`
under **"What you're given"**, and says *"Do not modify it."* Those files were
never received with the brief. This package is a hand-built replacement so the
agent runtime can actually be developed and tested against something.

It is named `mockllm_local`, not `mockllm`, specifically so it can never be
mistaken for the provided package or read as a modification of it. **The
graders run their own `mockllm/`.** Nothing here is a claim about how theirs
behaves — it is an implementation of the S1–S12 table in the brief, written
from that table alone.

## Running it

```
make serve                      # 127.0.0.1:8000, default scenario S1
make smoke                      # 19 checks that each scenario misbehaves as advertised
python -m mockllm_local --port 8100 --scenario S6
```

`make serve` seeds `workspace/` with the fixtures the scenarios read
(`notes.txt`, `injected_notes.txt`). Pass `--no-seed` to skip that.

## Endpoints

| Method | Path | Shape |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages — the shape the brief specifies |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions — for the Part B framework client |
| `GET` | `/health` | liveness + loaded scenario ids |
| `GET` | `/v1/scenarios` | scenario index with descriptions and turn counts |
| `GET` | `/v1/sessions` | current cursor per session, for debugging |
| `POST` | `/v1/sessions/reset` | reset one session (`X-Mock-Session`) or all |

Both endpoints are driven by the *same* scenario YAML, so a scenario behaves
identically whichever shape you ask for — that is asserted by the last smoke
check.

## Selecting a scenario

Highest precedence first:

1. `?scenario=S6` on the URL
2. `X-Mock-Scenario: S6` header
3. a model name ending in the id — `"model": "mock-s6"`, `"mockllm/S12"`
4. `MOCKLLM_SCENARIO=S6` in the environment
5. the server's `--scenario` default (`S1`)

Option 3 exists because some framework clients let you set a model name far
more easily than a custom header. Part B needs that.

## Sessions and the turn cursor

Each scenario is a flat list of turns. The server holds one cursor per session
and **advances it on every request**, including requests that are answered with
a fault. That is deliberate: it is what lets S6 script `429 → 529 → 200` even
though the client retries with a byte-identical body.

Session identity is:

- `X-Mock-Session: <anything>` if the client sends it, otherwise
- a SHA-256 of the first `user` message.

The fallback means an unmodified HTTP client that just retries lands in the
same session and walks the script forward, with no cooperation required.

State is in memory only and dies with the process. Restart the server, or
`POST /v1/sessions/reset`, to rewind.

## The scenarios

| ID | File | What it does to you |
|---|---|---|
| S1 | `s01_happy_path.yaml` | Control case: one tool call, clean finish. |
| S2 | `s02_malformed_arguments.yaml` | Trailing comma, raw newline inside a string, object truncated mid-key. |
| S3 | `s03_bad_tool_and_types.yaml` | Nonexistent tool; `path` as an int; `code` as an array; a required argument missing. |
| S4 | `s04_infinite_loop.yaml` | Same tool, same arguments, forever. Only R5 can stop it. |
| S5 | `s05_connection_reset.yaml` | Headers and part of the body land, then the socket is RST. Twice, then a good response. |
| S6 | `s06_rate_limit_then_overload.yaml` | `429` + `Retry-After: 1`, then `529` with no hint, then `200`. |
| S7 | `s07_prompt_injection.yaml` | Reads an injected file, then **complies**: exfil email, write to `../../.ssh/`, off-allow-list fetch. |
| S8 | `s08_context_blowup.yaml` | Response text doubles every turn, forever. Turn 0 states the fact turn 40 needs. |
| S9 | `s09_duplicate_tool_ids.yaml` | One `tool_use` id reused across four calls, two of them `send_email`. |
| S10 | `s10_parallel_fail_and_hang.yaml` | Three parallel calls: one succeeds, one errors, one never returns. |
| S11 | `s11_confidently_wrong.yaml` | Every call fails; the model narrates success and builds on invented content. |
| S12 | `s12_partial_turn.yaml` | Content-Length promises three tool calls, the body stops after the first, connection closes cleanly. |

### S5 vs S12 — they are not the same fault

S5 ends in a **reset**: `SO_LINGER 0`, so the peer gets an RST. Most clients
raise a connection error.

S12 ends in a clean **FIN** after a short write. The client sees a well-behaved
socket close and a body shorter than the `Content-Length` it was promised —
`http.client` raises `IncompleteRead`, and a client that reads leniently gets
unparseable JSON instead of an error. Handling one does not get you the other.

The cut offset is random but seeded on `(session, cursor)`, so a given session
reproduces the same offset. Hostile, but debuggable.

### S2 on two wire shapes

OpenAI carries tool arguments as a JSON *string*, so broken JSON survives
normal serialisation. Anthropic carries them as a JSON *object*, which cannot
hold unparseable text — so `wire.py` serialises a sentinel and substitutes the
raw text into the finished bytes. Same scenario, same brokenness, both shapes.

## The tokenizer

`mockllm_local/tokenizer.py` is deterministic and dependency-free: a GPT-2-style
greedy splitter with long words subdividing every 4 characters.

**It will not agree digit-for-digit with the real `mockllm/tokenizer.py`.** R3's
8,000 ceiling is graded against theirs, so treat the local number as a soft
target and keep headroom. `agent/memory.py` already budgets to 7,500.

## Known divergences from the real thing

- **No streaming.** Every response is a single non-streamed body. The real
  server may stream, which would change what "reset at a random byte offset"
  looks like on the client side.
- **Token counts differ** from the real tokenizer (above).
- **The redteam payloads are invented.** The brief says `harness/redteam/`
  contents are not disclosed. `mockllm_local/redteam/injected_notes.txt` is a
  plausible injection, not the real one — passing S7 here is evidence, not proof.
- **Sessions are in-memory**, so a restart rewinds every cursor.

`harness/chaos.py` is also hand-written — see the repo README.

## Addressed mode

A client that sends `X-Mock-Step` and `X-Mock-Attempt` gets a turn chosen as a
pure function of those two numbers instead of from a hidden cursor. It exists so
the chaos harness can assert an *exact* outcome: a run killed and resumed
re-requests step 4 and receives exactly the turn it received the first time.
Retries still walk forward within a step, so S6 behaves identically either way.
Without this, a kill landing between the server answering and the client
recording would silently shift the scenario by one turn, and a lost `send_email`
would be indistinguishable from a runtime bug.

## Not to be submitted as `mockllm/`

If the real package arrives, drop it in as `mockllm/` and this becomes dead
weight — `agent/memory.py` imports `mockllm.tokenizer` first and only falls
back to this one.
