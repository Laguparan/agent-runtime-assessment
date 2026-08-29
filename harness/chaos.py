"""R2: kill the agent at random points, resume it, and check the ledger.

The brief runs its own chaos harness 100 times and asserts `send_email` fired
exactly once per logical send. This is that test, written against our own
runtime so the guarantee is measured rather than asserted in a README.

One iteration:

    1. start `agent run` as a subprocess
    2. SIGKILL it after a random delay drawn from the observed run duration
    3. `agent resume` until it reaches a terminal stop reason (or the retry cap)
    4. check the invariants below

Invariants, in order of how much they would hurt to violate:

    NEVER TWICE   no two rows in `emails` share an idempotency key, and the
                  count for a run never exceeds the clean baseline
    NEVER ZERO    every send_email the log records as ok has exactly one row
    CONSISTENT    every emails row has an effects row and vice versa

The kill is a real SIGKILL. On Windows `Popen.kill()` is TerminateProcess, which
gives the child no chance to flush, clean up, or run an atexit hook -- which is
the point. Nothing in the runtime is allowed to depend on shutdown code running.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import storage  # noqa: E402
from mockllm_local.server import build_server, seed_workspace  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Its own port and its own server, so `make chaos` needs nothing else running
# and cannot disturb a mock server someone is using interactively.
CHAOS_PORT = 8766

# S9 issues two distinct sends to the same address. Two, not one: the model
# reuses a single tool_use id for both, so a runtime that keys idempotency on
# the model's id collapses them and fails this test with a count of 1.
DEFAULT_SCENARIO = "S9"
DEFAULT_TASK = "email both halves of the report to the team"
DEFAULT_RECIPIENT = "team@example.com"

MAX_RESUMES = 12


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["MOCKLLM_URL"] = f"http://127.0.0.1:{CHAOS_PORT}"
    return env


def _cli(args: list[str], db: str, traces: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agent.cli", "--db", db, "--traces", traces, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(),
    )


def _spawn(args: list[str], db: str, traces: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "agent.cli", "--db", db, "--traces", traces, *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_child_env(),
    )


def _wait_for_start(db: str, run_id: str, timeout: float) -> bool:
    """Block until the run's first event is committed, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = storage.connect(db)
        try:
            row = conn.execute(
                "SELECT 1 FROM events WHERE run_id = ? LIMIT 1", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            return True
        time.sleep(0.01)
    return False


def clean_baseline(db: str, traces: str, scenario: str, task: str, recipient: str) -> tuple[int, float]:
    """Run once without killing anything.

    Returns the expected email count and the run's *active* span -- first event
    to last, taken from the trace timestamps. Process wall-clock is the wrong
    number to aim kills at, because most of it is interpreter startup, and a
    kill that lands there tests nothing.
    """
    result = _cli(
        [
            "run", "--task", task, "--scenario", scenario,
            "--run-id", "baseline", "--allow-email", recipient, "--quiet",
        ],
        db, traces,
    )
    if result.returncode not in (0, 1):
        raise SystemExit(f"baseline run failed:\n{result.stdout}\n{result.stderr}")

    conn = storage.connect(db)
    try:
        expected = storage.count_emails(conn, "baseline")
        stamps = [
            event["created_at"] for event in storage.iter_events(conn, "baseline")
        ]
    finally:
        conn.close()

    active = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.1
    return expected, max(active, 0.05)


def one_iteration(
    index: int, db: str, traces: str, scenario: str, task: str, recipient: str, max_delay: float
) -> dict:
    run_id = f"chaos_{index:03d}"
    process = _spawn(
        [
            "run", "--task", task, "--scenario", scenario,
            "--run-id", run_id, "--allow-email", recipient, "--quiet",
        ],
        db, traces,
    )

    # Wait until the run has actually begun before starting the kill clock.
    # Killing during interpreter startup tests the Python launcher, not the
    # runtime, and leaves nothing to resume -- which would show up as a
    # "missing send" that is really just a run that never happened.
    started = _wait_for_start(db, run_id, timeout=20.0)
    if not started:
        return {"run_id": run_id, "never_started": True, "killed": False, "resumes": 0,
                "emails": 0, "distinct_keys": 0, "effect_rows": 0, "recorded_ok_sends": 0}

    # Uniform over the observed clean duration, so kills land across the whole
    # run rather than clustering at the start.
    time.sleep(random.uniform(0.0, max_delay))
    killed = process.poll() is None
    if killed:
        process.kill()
    process.wait(timeout=30)

    resumes = 0
    while resumes < MAX_RESUMES:
        result = _cli(["resume", run_id, "--quiet"], db, traces)
        resumes += 1
        # A resume that neither crashed nor was killed has reached a terminal
        # stop reason; returncode 0 is "completed", 1 is any other stop reason.
        if result.returncode in (0, 1):
            break

    conn = storage.connect(db)
    try:
        emails = storage.list_emails(conn, run_id)
        effects = conn.execute(
            "SELECT idempotency_key FROM effects WHERE run_id = ? AND tool = 'send_email'",
            (run_id,),
        ).fetchall()
        recorded_ok_sends = sum(
            1
            for event in storage.iter_events(conn, run_id)
            if event["kind"] == "tool_result"
            and event["payload"].get("tool") == "send_email"
            and event["payload"].get("ok")
            and not event["payload"].get("replayed")
        )
    finally:
        conn.close()

    keys = [row["idempotency_key"] for row in emails]
    return {
        "run_id": run_id,
        "killed": killed,
        "resumes": resumes,
        "emails": len(emails),
        "distinct_keys": len(set(keys)),
        "effect_rows": len(effects),
        "recorded_ok_sends": recorded_ok_sends,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chaos test for exactly-once side effects")
    parser.add_argument("-n", "--iterations", type=int, default=100)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--recipient", default=DEFAULT_RECIPIENT)
    parser.add_argument("--db", default=os.path.join(REPO_ROOT, "chaos_state.db"))
    parser.add_argument("--traces", default=os.path.join(REPO_ROOT, "chaos_traces"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep", action="store_true", help="Keep the chaos db and traces.")
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    for path in (args.db, args.db + "-wal", args.db + "-shm"):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(args.traces, ignore_errors=True)
    storage.init_db(args.db)

    seed_workspace(os.path.join(REPO_ROOT, "workspace"))
    server = build_server("127.0.0.1", CHAOS_PORT, DEFAULT_SCENARIO, verbose=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"mock server on 127.0.0.1:{CHAOS_PORT}")

    try:
        return _run(args, server)
    finally:
        server.shutdown()
        server.server_close()


def _run(args, server) -> int:
    expected, active = clean_baseline(
        args.db, args.traces, args.scenario, args.task, args.recipient
    )
    print(
        f"baseline: {args.scenario} sends {expected} email(s); the run is active "
        f"for {active * 1000:.0f}ms from its first event to its last\n"
    )
    if expected == 0:
        raise SystemExit(
            "baseline sent no emails, so this test would prove nothing. "
            "Check that the scenario and --recipient agree."
        )

    # Slightly wider than the active span, so some kills land after the run
    # finished (which must also be handled correctly) but most land inside it.
    max_delay = active * 1.1
    results = []
    failures = []

    for index in range(args.iterations):
        result = one_iteration(
            index, args.db, args.traces, args.scenario, args.task, args.recipient, max_delay
        )
        results.append(result)

        if result.get("never_started"):
            print(f"  [{index:3d}] SKIP {result['run_id']}: never reached its first event")
            continue

        problems = []
        if result["emails"] > expected:
            problems.append(f"DUPLICATE SEND: {result['emails']} emails, baseline is {expected}")
        if result["emails"] != result["distinct_keys"]:
            problems.append("two emails share an idempotency key")
        if result["emails"] != result["effect_rows"]:
            problems.append(
                f"ledger disagrees with outbox: {result['effect_rows']} effects, "
                f"{result['emails']} emails"
            )
        if result["emails"] < expected and result["resumes"] < MAX_RESUMES:
            problems.append(f"MISSING SEND: {result['emails']} emails, baseline is {expected}")

        if problems:
            failures.append((result, problems))
            print(f"  [{index:3d}] FAIL {result['run_id']}: {'; '.join(problems)}")
        elif (index + 1) % 10 == 0:
            killed = sum(1 for r in results if r["killed"])
            print(
                f"  [{index + 1:3d}/{args.iterations}] ok "
                f"({killed} killed mid-run so far, "
                f"{sum(r['resumes'] for r in results)} resumes)"
            )

    scored = [r for r in results if not r.get("never_started")]
    killed = sum(1 for r in scored if r["killed"])
    counts: dict[int, int] = {}
    for result in scored:
        counts[result["emails"]] = counts.get(result["emails"], 0) + 1

    print("\n" + "=" * 64)
    print(f"iterations           {args.iterations} ({len(scored)} scored)")
    print(f"killed mid-run       {killed}")
    print(f"total resumes        {sum(r['resumes'] for r in scored)}")
    print(f"expected per run     {expected}")
    print(f"observed email count {json.dumps({str(k): v for k, v in sorted(counts.items())})}")
    print(f"failures             {len(failures)}")
    print("=" * 64)

    if failures:
        print("\nexactly-once was VIOLATED:")
        for result, problems in failures[:10]:
            print(f"  {result['run_id']}: {'; '.join(problems)}")
        return 1

    if killed == 0:
        print("\nno iteration was actually killed mid-run -- this proved nothing.")
        print("the runs are finishing faster than the kill delay; lower --max-delay.")
        return 1

    print(f"\nexactly-once held across {args.iterations} iterations ({killed} killed mid-run).")
    if not args.keep:
        for path in (args.db, args.db + "-wal", args.db + "-shm"):
            if os.path.exists(path):
                os.remove(path)
        shutil.rmtree(args.traces, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
