"""R6: structured JSONL traces.

Every run appends to `traces/<run_id>.jsonl`, one JSON object per line, never
rewritten. Each line is flushed and fsynced before the call returns, so a trace
tells the truth about a process that was killed a microsecond later -- a trace
that loses its last few lines to a buffer is worse than no trace, because it
lies about where the run got to.

The trace and the SQLite event log carry the same events. SQLite is what the
runtime reads to resume; the JSONL is what a human (or `agent replay`) reads.
They are written from the same call site so they cannot drift.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator

from . import config


class Tracer:
    def __init__(self, run_id: str, trace_dir: str | None = None) -> None:
        self.run_id = run_id
        directory = trace_dir or config.TRACE_DIR
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{run_id}.jsonl")
        self._handle = open(self.path, "a", encoding="utf-8")

    def emit(self, step: int, kind: str, payload: dict[str, Any]) -> None:
        line = json.dumps(
            {
                "ts": round(time.time(), 6),
                "run_id": self.run_id,
                "step": step,
                "kind": kind,
                **payload,
            },
            default=str,
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        # Durability, not tidiness: without this the last lines vanish on kill -9.
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_trace(run_id: str, trace_dir: str | None = None) -> Iterator[dict[str, Any]]:
    path = os.path.join(trace_dir or config.TRACE_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trace for run {run_id} at {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A half-written final line is the normal signature of kill -9.
                # Stop cleanly rather than pretending the trace is corrupt.
                print(f"[trace] {path}: line {number} is truncated; stopping there.")
                return
