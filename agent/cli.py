"""Command line entry point.

    python -m agent.cli run --task "..." [--allow-email addr] [--scenario S7]
    python -m agent.cli resume <run_id>
    python -m agent.cli replay <run_id>
    python -m agent.cli inspect <run_id>
    python -m agent.cli emails

`send_email` is off unless `--allow-email` names a recipient. That is not a
convenience flag -- it is the R4 trust boundary, and it is why the S7 injection
fails against a default run. See policy.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from . import config, storage
from .loop import StopReason, run_agent
from .policy import RunPolicy
from .replay import replay
from .trace import read_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="Adversarial agent runtime")
    parser.add_argument("--db", default=None, help="SQLite path (default: agent_state.db)")
    parser.add_argument("--traces", default=None, help="Trace directory (default: traces/)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Start a new run")
    run.add_argument("--task", required=True)
    run.add_argument("--run-id", default=None)
    run.add_argument("--scenario", default=None, help="Mock scenario to request, e.g. S7")
    run.add_argument(
        "--allow-email",
        action="append",
        metavar="ADDRESS",
        help="Authorise send_email to this address. Repeatable. Omit and send_email is refused.",
    )
    run.add_argument("--allow-host", action="append", metavar="HOST", help="Override the http_get allow-list.")
    run.add_argument("--no-write", action="store_true")
    run.add_argument("--no-python", action="store_true")
    run.add_argument("--quiet", action="store_true")

    resume = sub.add_parser("resume", help="Continue a run after a crash")
    resume.add_argument("run_id")
    resume.add_argument("--quiet", action="store_true")

    replay_parser = sub.add_parser("replay", help="Re-derive decisions from the trace, no server")
    replay_parser.add_argument("run_id")

    inspect = sub.add_parser("inspect", help="Print a run's trace")
    inspect.add_argument("run_id")
    inspect.add_argument("--kind", default=None, help="Only show events of this kind")

    sub.add_parser("emails", help="List every email the ledger says was sent")
    sub.add_parser("runs", help="List recent runs")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
        policy = RunPolicy.from_args(args)
        try:
            outcome = run_agent(
                run_id,
                args.task,
                policy=policy,
                scenario=args.scenario,
                db_path=args.db,
                trace_dir=args.traces,
                verbose=not args.quiet,
            )
        except KeyboardInterrupt:
            print(f"\ninterrupted. resume with: python -m agent.cli resume {run_id}")
            return 130
        return 0 if outcome.stop_reason is StopReason.COMPLETED else 1

    if args.command == "resume":
        outcome = run_agent(
            args.run_id,
            task="",
            resume=True,
            db_path=args.db,
            trace_dir=args.traces,
            verbose=not args.quiet,
        )
        return 0 if outcome.stop_reason is StopReason.COMPLETED else 1

    if args.command == "replay":
        report = replay(args.run_id, trace_dir=args.traces)
        return 0 if report.faithful else 1

    if args.command == "inspect":
        for event in read_trace(args.run_id, args.traces):
            if args.kind and event["kind"] != args.kind:
                continue
            print(json.dumps(event, indent=2, default=str))
        return 0

    if args.command == "emails":
        storage.init_db(args.db)
        conn = storage.connect(args.db)
        try:
            rows = storage.list_emails(conn)
            if not rows:
                print("no emails have been sent.")
            for row in rows:
                print(
                    f"#{row['id']} run={row['run_id']} to={row['to_addr']} "
                    f"subject={row['subject']!r} key={row['idempotency_key'][:12]}"
                )
            print(f"\n{len(rows)} email(s) total.")
        finally:
            conn.close()
        return 0

    if args.command == "runs":
        storage.init_db(args.db)
        conn = storage.connect(args.db)
        try:
            for row in storage.list_runs(conn):
                print(f"{row['run_id']}  scenario={row['scenario'] or '-':<5}  {row['task'][:60]}")
        finally:
            conn.close()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
