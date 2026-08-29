"""Smoke test for the mock server. Proves each scenario misbehaves as advertised.

Run the server first (`make serve`), then `python -m mockllm_local.smoke_test`.
This tests the *server*, not the agent -- the agent's own suite lives in evals/.
"""

from __future__ import annotations

import http.client
import json
import sys
import uuid

HOST = "127.0.0.1"
PORT = 8000
TIMEOUT = 8.0


class Outcome:
    """What one request did: a parsed body, or the way it broke."""

    def __init__(self, status=None, headers=None, body=None, failure=None):
        self.status = status
        self.headers = headers or {}
        self.body = body
        self.failure = failure

    def __repr__(self) -> str:
        if self.failure:
            return f"<{self.failure}>"
        return f"<{self.status} {str(self.body)[:60]}>"


def post(path: str, payload: dict, session: str, scenario: str | None = None) -> Outcome:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=TIMEOUT)
    headers = {"Content-Type": "application/json", "X-Mock-Session": session}
    if scenario:
        headers["X-Mock-Scenario"] = scenario
    try:
        conn.request("POST", path, json.dumps(payload), headers)
        response = conn.getresponse()
        raw = response.read()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return Outcome(response.status, dict(response.getheaders()), failure="unparseable-body")
        return Outcome(response.status, dict(response.getheaders()), body)
    except http.client.IncompleteRead:
        return Outcome(failure="incomplete-read")
    except (ConnectionResetError, ConnectionAbortedError):
        return Outcome(failure="connection-reset")
    except OSError as exc:
        return Outcome(failure=f"os-error:{exc.__class__.__name__}")
    finally:
        conn.close()


def turn(path: str, scenario: str, session: str, text: str = "do the task") -> Outcome:
    payload = {
        "model": "mock-model",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": text}],
    }
    return post(path, payload, session, scenario)


CHECKS: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, "PASS" if ok else "FAIL"))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))


def _unparseable(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return True
    return False


def anthropic_blocks(outcome: Outcome, block_type: str) -> list[dict]:
    if not isinstance(outcome.body, dict):
        return []
    return [b for b in outcome.body.get("content", []) if b.get("type") == block_type]


def main() -> int:
    messages = "/v1/messages"
    completions = "/v1/chat/completions"

    print("\nS1  happy path")
    sid = uuid.uuid4().hex
    first = turn(messages, "S1", sid)
    second = turn(messages, "S1", sid)
    check("first turn returns one tool_use", len(anthropic_blocks(first, "tool_use")) == 1)
    check("second turn ends the conversation",
          isinstance(second.body, dict) and second.body.get("stop_reason") == "end_turn")

    print("\nS2  malformed arguments reach the client unparseable")
    sid = uuid.uuid4().hex
    broken = 0
    for _ in range(3):
        out = turn(completions, "S2", sid)
        for call in out.body["choices"][0]["message"].get("tool_calls", []):
            try:
                json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                broken += 1
    check("3 of the first 3 turns carry unparseable arguments", broken == 3, f"{broken}/3")

    sid = uuid.uuid4().hex
    block = anthropic_blocks(turn(messages, "S2", sid), "tool_use")[0]
    check("Anthropic shape carries raw broken text as a string input",
          isinstance(block["input"], str) and _unparseable(block["input"]),
          repr(block["input"])[:60])

    print("\nS3  unknown tool and wrong-typed arguments")
    sid = uuid.uuid4().hex
    names = [anthropic_blocks(turn(messages, "S3", sid), "tool_use") for _ in range(2)]
    check("turn 0 calls a tool that does not exist",
          names[0] and names[0][0]["name"] == "delete_directory")

    print("\nS4  never terminates on its own")
    sid = uuid.uuid4().hex
    ids = {anthropic_blocks(turn(messages, "S4", sid), "tool_use")[0]["id"] for _ in range(6)}
    check("6 requests, always the same tool_use id", ids == {"toolu_s4_001"})

    print("\nS5  connection reset mid-response")
    sid = uuid.uuid4().hex
    outcomes = [turn(messages, "S5", sid) for _ in range(3)]
    failures = [o.failure for o in outcomes[:2]]
    check("first two requests break mid-body",
          all(f in ("incomplete-read", "connection-reset") for f in failures), str(failures))
    check("third request succeeds", outcomes[2].status == 200)

    print("\nS6  429 -> 529 -> 200")
    sid = uuid.uuid4().hex
    a, b, c = (turn(messages, "S6", sid) for _ in range(3))
    check("429 first with Retry-After",
          a.status == 429 and a.headers.get("Retry-After") == "1")
    check("529 second", b.status == 529)
    check("200 third", c.status == 200)

    print("\nS7  model complies with the injection")
    sid = uuid.uuid4().hex
    turn(messages, "S7", sid)
    exfil = anthropic_blocks(turn(messages, "S7", sid), "tool_use")
    check("turn 1 attempts send_email to an attacker address",
          exfil and exfil[0]["name"] == "send_email"
          and "attacker" in exfil[0]["input"]["to"])

    print("\nS8  responses grow without bound")
    sid = uuid.uuid4().hex
    sizes = []
    for _ in range(5):
        out = turn(messages, "S8", sid)
        sizes.append(len(json.dumps(out.body)))
    check("each response past the script is larger than the last",
          all(x < y for x, y in zip(sizes[1:], sizes[2:])), str(sizes))

    print("\nS9  duplicate tool_use ids")
    sid = uuid.uuid4().hex
    seen = [anthropic_blocks(turn(messages, "S9", sid), "tool_use")[0]["id"] for _ in range(4)]
    check("4 distinct calls share one id", len(set(seen)) == 1, seen[0])

    print("\nS10 parallel calls, one fails, one hangs")
    sid = uuid.uuid4().hex
    blocks = anthropic_blocks(turn(messages, "S10", sid), "tool_use")
    check("one turn carries three tool_use blocks", len(blocks) == 3, str(len(blocks)))

    print("\nS11 confidently wrong")
    sid = uuid.uuid4().hex
    turn(messages, "S11", sid)
    claim = turn(messages, "S11", sid)
    check("claims success on a read that failed",
          "successfully" in anthropic_blocks(claim, "text")[0]["text"])

    print("\nS12 partial interrupted turn")
    sid = uuid.uuid4().hex
    cut = turn(messages, "S12", sid)
    full = turn(messages, "S12", sid)
    check("first request delivers an unparseable prefix",
          cut.failure in ("incomplete-read", "unparseable-body"), str(cut.failure))
    check("retry delivers all three calls", len(anthropic_blocks(full, "tool_use")) == 3)

    print("\nBoth wire shapes agree on the script")
    sid_a, sid_o = uuid.uuid4().hex, uuid.uuid4().hex
    anth = anthropic_blocks(turn(messages, "S1", sid_a), "tool_use")[0]
    oai = turn(completions, "S1", sid_o).body["choices"][0]["message"]["tool_calls"][0]
    check("same tool, same id, same arguments",
          anth["name"] == oai["function"]["name"]
          and anth["id"] == oai["id"]
          and anth["input"] == json.loads(oai["function"]["arguments"]))

    failed = [name for name, status in CHECKS if status == "FAIL"]
    print("\n" + "=" * 60)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConnectionRefusedError, OSError) as exc:
        print(f"Cannot reach the mock server on {HOST}:{PORT} ({exc}).")
        print("Start it with:  make serve")
        sys.exit(2)
