"""The hostile local mock model server.

Serves one scripted scenario per session over two endpoints:

    POST /v1/messages            Anthropic Messages shape (the brief's shape)
    POST /v1/chat/completions    OpenAI shape (for the Part B framework client)

Scenario selection, highest precedence first:

    ?scenario=S6                 query string
    X-Mock-Scenario: S6          request header
    "model": "mock-s6"           model name suffix
    MOCKLLM_SCENARIO=S6          environment
    S1                           default

Session identity decides which cursor a request advances. A client that sends
`X-Mock-Session` controls its own; anything else is keyed on a hash of the first
user message, so a retry of an identical request lands in the same session and
walks the script forward. That is what makes S6's 429 -> 529 -> 200 terminate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import scenario as scenario_mod
from . import wire
from .scenario import Scenario, Turn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_SCENARIO = "S1"

_ENDPOINT_SHAPES = {
    "/v1/messages": wire.SHAPE_ANTHROPIC,
    "/v1/chat/completions": wire.SHAPE_OPENAI,
}


class SessionStore:
    """Per-session turn selection.

    Two modes, and the difference matters for testing under chaos.

    *Cursor mode* (no `X-Mock-Step`): every request advances a hidden counter.
    Simple, and enough for a client that just retries.

    *Addressed mode* (`X-Mock-Step` and `X-Mock-Attempt` present): the turn
    served is a pure function of (step, attempt). A client that crashes and
    re-requests step 4 gets exactly the turn it got the first time, so a chaos
    harness can assert an exact outcome instead of a distribution. Retries still
    walk forward -- attempt 1 is the next turn after attempt 0 -- so S6's
    `429 -> 529 -> 200` behaves identically in both modes.
    """

    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}
        self._step_starts: dict[str, dict[int, int]] = {}
        self._next_start: dict[str, int] = {}
        self._lock = threading.Lock()

    def advance(self, key: str) -> int:
        """Cursor mode: return the current position and move on by one."""
        with self._lock:
            cursor = self._cursors.get(key, 0)
            self._cursors[key] = cursor + 1
            return cursor

    def address(self, key: str, step: int, attempt: int) -> int:
        """Addressed mode: the turn for (step, attempt), stable across restarts."""
        with self._lock:
            starts = self._step_starts.setdefault(key, {})
            if step not in starts:
                starts[step] = self._next_start.get(key, 0)
            cursor = starts[step] + attempt
            self._next_start[key] = max(self._next_start.get(key, 0), cursor + 1)
            self._cursors[key] = cursor + 1
            return cursor

    def reset(self, key: str | None = None) -> int:
        with self._lock:
            if key is None:
                count = len(self._cursors)
                self._cursors.clear()
                self._step_starts.clear()
                self._next_start.clear()
                return count
            self._step_starts.pop(key, None)
            self._next_start.pop(key, None)
            return 1 if self._cursors.pop(key, None) is not None else 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._cursors)


class MockLLMHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 so Content-Length is honoured and the reset/truncate faults
    # actually look like a broken response rather than a normal EOF.
    protocol_version = "HTTP/1.1"
    server_version = "mockllm-local/1.0"

    # Injected by build_server().
    scenarios: dict[str, Scenario] = {}
    sessions: SessionStore = SessionStore()
    default_scenario: str = DEFAULT_SCENARIO
    verbose: bool = True

    # ---------------------------------------------------------------- routing

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "scenarios": sorted(self.scenarios)})
        elif path == "/v1/scenarios":
            self._send_json(200, {"scenarios": self._scenario_index()})
        elif path == "/v1/sessions":
            self._send_json(200, {"sessions": self.sessions.snapshot()})
        else:
            self._send_json(404, {"error": f"no route for GET {path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]

        if path == "/v1/sessions/reset":
            key = self.headers.get("X-Mock-Session")
            cleared = self.sessions.reset(key)
            self._send_json(200, {"cleared": cleared})
            return

        shape = _ENDPOINT_SHAPES.get(path)
        if shape is None:
            self._send_json(404, {"error": f"no route for POST {path}"})
            return

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_error(400, shape, str(exc))
            return

        scenario = self._resolve_scenario(payload)
        if scenario is None:
            self._send_error(400, shape, "unknown scenario; GET /v1/scenarios to list")
            return

        session_key = self._session_key(scenario, payload)
        step = _int_header(self.headers.get("X-Mock-Step"))
        attempt = _int_header(self.headers.get("X-Mock-Attempt"))
        if step is None:
            cursor = self.sessions.advance(session_key)
        else:
            cursor = self.sessions.address(session_key, step, attempt or 0)
        turn = scenario.turn_at(cursor)

        if turn is None:
            # The script ran out. Say so plainly rather than looping silently --
            # a scenario running off its end is a bug in the test, not a fault
            # the agent is supposed to survive.
            self._send_error(
                409,
                shape,
                f"{scenario.id} has only {len(scenario.turns)} turns; "
                f"request {cursor + 1} is past the end of the script",
            )
            return

        self._serve_turn(turn, scenario, shape, payload, session_key, cursor)

    # ------------------------------------------------------------ turn faults

    def _serve_turn(
        self,
        turn: Turn,
        scenario: Scenario,
        shape: str,
        payload: dict[str, Any],
        session_key: str,
        cursor: int,
    ) -> None:
        self._log(f"{scenario.id} turn {cursor} -> {turn.kind} [{session_key[:12]}]")

        if turn.kind == scenario_mod.KIND_HTTP_ERROR:
            self._send_error(
                turn.status,
                shape,
                f"{scenario.id}: scripted {turn.status}",
                extra_headers=turn.headers,
            )
            return

        if turn.kind == scenario_mod.KIND_HANG:
            time.sleep(turn.seconds)
            if turn.then_drop:
                self._hard_reset()
                return
            # Fall through and answer late.

        model = str(payload.get("model") or "mock-model")
        messages = payload.get("messages") or []
        body = wire.render(turn, scenario, shape, model, messages, cursor=cursor)

        if turn.kind == scenario_mod.KIND_RESET:
            self._send_partial(body, session_key, cursor, reset=True)
        elif turn.kind == scenario_mod.KIND_TRUNCATE:
            self._send_partial(body, session_key, cursor, reset=False)
        else:
            self._send_bytes(200, body)

    def _send_partial(
        self, body: bytes, session_key: str, cursor: int, *, reset: bool
    ) -> None:
        """Promise the whole body, deliver a prefix of it, then break off.

        The cut point is random but seeded on (session, cursor), so a given
        session reproduces the same offset -- random enough to be hostile,
        deterministic enough to debug.
        """
        rng = random.Random(f"{session_key}:{cursor}")
        # Always past the headers and always short of the end, so the client
        # sees a genuinely partial document rather than an empty or whole one.
        offset = rng.randint(max(1, len(body) // 10), max(2, (len(body) * 4) // 5))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))  # deliberately a lie
        self.end_headers()

        self.wfile.write(body[:offset])
        self.wfile.flush()
        self._log(f"  delivered {offset}/{len(body)} bytes, then {'RST' if reset else 'FIN'}")

        if reset:
            self._hard_reset()
        else:
            self.close_connection = True

    def _hard_reset(self) -> None:
        """Close with SO_LINGER 0 so the peer gets an RST, not a clean FIN."""
        try:
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.connection.close()
        except OSError:
            pass
        self.close_connection = True

    # -------------------------------------------------------------- selection

    def _resolve_scenario(self, payload: dict[str, Any]) -> Scenario | None:
        candidates = [
            _query_param(self.path, "scenario"),
            self.headers.get("X-Mock-Scenario"),
            _scenario_from_model(payload.get("model")),
            os.environ.get("MOCKLLM_SCENARIO"),
            self.default_scenario,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            scenario = self.scenarios.get(str(candidate).strip().upper())
            if scenario is not None:
                return scenario
        return None

    def _session_key(self, scenario: Scenario, payload: dict[str, Any]) -> str:
        explicit = self.headers.get("X-Mock-Session")
        if explicit:
            return f"{scenario.id}:{explicit}"

        # Fall back to the first user message, so retries of an identical
        # request stay in the same session without the client cooperating.
        seed = ""
        for message in payload.get("messages") or []:
            if message.get("role") == "user":
                seed = json.dumps(message.get("content"), sort_keys=True)
                break
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return f"{scenario.id}:auto-{digest}"

    def _scenario_index(self) -> list[dict[str, Any]]:
        return [
            {
                "id": scn.id,
                "name": scn.name,
                "description": scn.description.strip(),
                "turns": len(scn.turns),
                "repeats_last_turn": scn.repeat_last,
            }
            for scn in sorted(self.scenarios.values(), key=lambda s: _sort_key(s.id))
        ]

    # ------------------------------------------------------------------- I/O

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise ValueError("Content-Length is not an integer") from None
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"request body is not valid JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_bytes(
        self, status: int, body: bytes, extra_headers: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        self._send_bytes(status, json.dumps(payload, indent=2).encode("utf-8"))

    def _send_error(
        self,
        status: int,
        shape: str,
        message: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._send_bytes(status, wire.error_body(shape, status, message), extra_headers)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[mockllm] {message}", flush=True)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Silence the default per-request access log; _log is more useful.
        return


def _int_header(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _query_param(path: str, name: str) -> str | None:
    if "?" not in path:
        return None
    from urllib.parse import parse_qs

    values = parse_qs(path.split("?", 1)[1]).get(name)
    return values[0] if values else None


def _scenario_from_model(model: Any) -> str | None:
    """Pull a scenario id out of a model name like `mock-s6` or `mockllm/S12`."""
    if not isinstance(model, str):
        return None
    tail = model.replace("/", "-").rsplit("-", 1)[-1].strip().upper()
    if len(tail) >= 2 and tail[0] == "S" and tail[1:].isdigit():
        return tail
    return None


def _sort_key(scenario_id: str) -> tuple[int, str]:
    digits = scenario_id.lstrip("Ss")
    return (int(digits), scenario_id) if digits.isdigit() else (9999, scenario_id)


def seed_workspace(workspace: str) -> list[str]:
    """Drop the files the scenarios expect to read into `workspace`.

    The real assessment mounts `harness/redteam/` for this. Those payloads were
    not provided, so `mockllm_local/redteam/` carries stand-ins.
    """
    import pathlib
    import shutil

    target = pathlib.Path(workspace)
    target.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(__file__).parent / "redteam"

    written = []
    for path in sorted(source.glob("*.txt")):
        shutil.copyfile(path, target / path.name)
        written.append(path.name)
    return written


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """Refuse to start if the port is taken.

    ThreadingHTTPServer defaults to `allow_reuse_address = 1`. On Windows that
    is not the POSIX TIME_WAIT convenience it is on Linux -- SO_REUSEADDR there
    lets a second process bind a port that is actively in use, and the OS then
    splits incoming connections between the two. Two mock servers on 8000 with
    independent turn cursors is a debugging nightmare that looks like flaky
    scenarios. Failing to bind is far better than answering half the requests.
    """

    allow_reuse_address = False
    daemon_threads = True


def build_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    default_scenario: str = DEFAULT_SCENARIO,
    verbose: bool = True,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundMockLLMHandler",
        (MockLLMHandler,),
        {
            "scenarios": scenario_mod.load_all(),
            "sessions": SessionStore(),
            "default_scenario": default_scenario.upper(),
            "verbose": verbose,
        },
    )
    try:
        return _ExclusiveHTTPServer((host, port), handler)
    except OSError as exc:
        raise SystemExit(
            f"[mockllm] cannot bind {host}:{port} ({exc}). "
            f"Another mock server is probably already running -- stop it, or "
            f"pass --port."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mockllm_local",
        description="Local hostile mock model server (stand-in for mockllm/).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--scenario",
        default=os.environ.get("MOCKLLM_SCENARIO", DEFAULT_SCENARIO),
        help="Scenario served when a request does not select one (default: S1).",
    )
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Directory to seed with scenario fixtures (default: workspace).",
    )
    parser.add_argument("--no-seed", action="store_true", help="Skip seeding fixtures.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_seed:
        seeded = seed_workspace(args.workspace)
        if seeded and not args.quiet:
            print(f"[mockllm] seeded {args.workspace}/: {', '.join(seeded)}")

    server = build_server(args.host, args.port, args.scenario, verbose=not args.quiet)
    count = len(server.RequestHandlerClass.scenarios)  # type: ignore[attr-defined]
    print(
        f"[mockllm] {count} scenarios loaded, default {args.scenario.upper()}\n"
        f"[mockllm] POST http://{args.host}:{args.port}/v1/messages          (Anthropic shape)\n"
        f"[mockllm] POST http://{args.host}:{args.port}/v1/chat/completions  (OpenAI shape)\n"
        f"[mockllm] GET  http://{args.host}:{args.port}/v1/scenarios\n"
        f"[mockllm] Ctrl-C to stop.",
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mockllm] shutting down.")
    finally:
        server.server_close()
    return 0
