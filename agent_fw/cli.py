"""Command line entry point for the framework build.

    python -m agent_fw.cli run --task "..." [--allow-email addr] [--scenario S7]
    python -m agent_fw.cli resume <run_id>
    python -m agent_fw.cli inspect <run_id>
    python -m agent_fw.cli emails

Deliberately the same shape as `agent.cli`, so the chaos harness and the evals
can drive either build by changing one module name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from typing import Any

from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage

from agent import storage
from agent.policy import RunPolicy
from agent.trace import Tracer, read_trace

from .deps import FwDeps
from .durability import load_history, persist_history
from .runtime import NoProgress, build_agent, usage_limits
from .telemetry import report, summary


async def _drive(deps: FwDeps, task: str, scenario: str | None, resume: bool) -> dict[str, Any]:
    """One run. Returns the F4 extraction regardless of how it ended."""
    agent = build_agent(deps, scenario)
    history = load_history(deps.conn, deps.run_id) if resume else None

    if resume and history:
        print(f"resuming {deps.run_id} from {len(history)} persisted messages")

    stop_reason, detail, output = "completed", "the model finished", None

    # Passed in rather than read off the result, because a run that raises has
    # no result to read. S4 stops inside a hook and reported 0 tokens and 0
    # steps until this was threaded through -- the framework only hands usage
    # back on the success path, and every interesting stop is a failure path.
    usage = RunUsage()

    try:
        result = await agent.run(
            None if (resume and history) else task,
            message_history=history,
            deps=deps,
            usage_limits=usage_limits(),
            usage=usage,
        )
        output = result.output
        persist_history(deps, result.all_messages())

    except UsageLimitExceeded as exc:
        # The framework's own ceiling. R5, enforced by a primitive.
        stop_reason, detail = "budget_exceeded", str(exc)
    except NoProgress as exc:
        # Our hook, because the framework has no equivalent.
        stop_reason, detail = "no_progress", str(exc)
    except UnexpectedModelBehavior as exc:
        stop_reason, detail = "model_behaviour", str(exc)
    except ModelHTTPError as exc:
        stop_reason, detail = "model_unavailable", str(exc)
    except Exception as exc:  # noqa: BLE001 - a run still owes a stop reason
        stop_reason, detail = "runtime_error", f"{exc.__class__.__name__}: {exc}"

    data = report(deps, usage, output, stop_reason, detail)
    deps.record(data["steps"], "run_ended", data)
    return data


def _run(args: argparse.Namespace, run_id: str, resume: bool) -> int:
    storage.init_db(args.db)
    conn = storage.connect(args.db)
    try:
        if resume:
            row = storage.get_run(conn, run_id)
            if row is None:
                raise SystemExit(f"no run named {run_id} to resume")
            task, scenario = row["task"], row["scenario"]
            policy = RunPolicy.from_dict(json.loads(row["policy_json"]))
        else:
            task, scenario = args.task, args.scenario
            policy = RunPolicy.from_args(args)
            if storage.get_run(conn, run_id) is not None:
                raise SystemExit(
                    f"run {run_id} already exists; use 'resume' or pick another --run-id"
                )
            storage.create_run(conn, run_id, task, policy.to_dict(), scenario)

        with Tracer(run_id, args.traces) as tracer:
            deps = FwDeps(run_id=run_id, policy=policy, conn=conn, tracer=tracer)
            if not resume:
                deps.record(0, "run_started", {"task": task, "policy": policy.to_dict(),
                                               "scenario": scenario, "build": "agent_fw"})
            data = asyncio.run(_drive(deps, task, scenario, resume))

        if not args.quiet:
            print(summary(data, storage.count_emails(conn, run_id)))
        return 0 if data["stop_reason"] == "completed" else 1
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_fw", description="Part B: Pydantic AI build")
    parser.add_argument("--db", default=None)
    parser.add_argument("--traces", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--run-id", default=None)
    run.add_argument("--scenario", default=None)
    run.add_argument("--allow-email", action="append", metavar="ADDRESS")
    run.add_argument("--allow-host", action="append", metavar="HOST")
    run.add_argument("--no-write", action="store_true")
    run.add_argument("--no-python", action="store_true")
    run.add_argument("--quiet", action="store_true")

    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--quiet", action="store_true")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("run_id")
    inspect.add_argument("--kind", default=None)

    sub.add_parser("emails")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        return _run(args, args.run_id or f"fw_{uuid.uuid4().hex[:8]}", resume=False)

    if args.command == "resume":
        args.task = args.scenario = None
        return _run(args, args.run_id, resume=True)

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
            for row in rows:
                print(
                    f"#{row['id']} run={row['run_id']} to={row['to_addr']} "
                    f"subject={row['subject']!r} key={row['idempotency_key'][:12]}"
                )
            print(f"\n{len(rows)} email(s) total.")
        finally:
            conn.close()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
