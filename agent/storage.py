"""SQLite durability layer: append-only event log plus the effect ledger.

The exactly-once guarantee in R2 rests on one idea. `send_email`'s observable
effect is a row in `emails`, and the ledger entry that claims that send is a row
in `effects` with a PRIMARY KEY on the idempotency key. Both rows are written in
a **single transaction**, so SQLite's own durability decides the outcome: either
the send is recorded and ledgered, or neither row exists. There is no window in
which a `kill -9` leaves the effect performed but unledgered.

`PRAGMA synchronous=FULL` is what makes that true across a power cut as well as
a process kill. It costs an fsync per commit; that is the price of the guarantee
and it is paid deliberately.

If the effect were a real SMTP call it could not live inside the transaction,
and this design would have to become a proper outbox: commit the intent, send,
commit the completion, and reconcile the PENDING rows on startup using the
provider's own idempotency key. See DECISIONS.md.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    scenario    TEXT,
    created_at  REAL NOT NULL
);

-- R2: append-only. Rows are only ever INSERTed; nothing updates or deletes.
-- Every projection the runtime needs (message history, step number, which
-- tools already ran) is rebuilt by folding this log.
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    step       INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id, seq);

-- R2: one row per logical irreversible effect. The PRIMARY KEY is the guarantee.
CREATE TABLE IF NOT EXISTS effects (
    idempotency_key TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    step            INTEGER NOT NULL,
    tool            TEXT NOT NULL,
    args_json       TEXT NOT NULL,
    result          TEXT NOT NULL,
    created_at      REAL NOT NULL
);

-- The delivered side effect. Written in the same transaction as its effects row.
CREATE TABLE IF NOT EXISTS emails (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    run_id          TEXT NOT NULL,
    to_addr         TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    sent_at         REAL NOT NULL
);
"""


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL so the chaos harness can read while a run writes.
    conn.execute("PRAGMA journal_mode=WAL")
    # FULL, not NORMAL: an fsync per commit is what survives a hard kill.
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | None = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


# --------------------------------------------------------------------- runs


def create_run(
    conn: sqlite3.Connection,
    run_id: str,
    task: str,
    policy: dict[str, Any],
    scenario: str | None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, task, policy_json, scenario, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, task, json.dumps(policy, sort_keys=True), scenario, time.time()),
    )


def get_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()


# ------------------------------------------------------------------- events


def append_event(
    conn: sqlite3.Connection, run_id: str, step: int, kind: str, payload: dict[str, Any]
) -> int:
    cursor = conn.execute(
        "INSERT INTO events (run_id, step, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, step, kind, json.dumps(payload, default=str), time.time()),
    )
    return int(cursor.lastrowid)


def iter_events(conn: sqlite3.Connection, run_id: str) -> Iterator[dict[str, Any]]:
    rows = conn.execute(
        "SELECT seq, step, kind, payload, created_at FROM events "
        "WHERE run_id = ? ORDER BY seq",
        (run_id,),
    )
    for row in rows:
        yield {
            "seq": row["seq"],
            "step": row["step"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload"]),
        }


# ------------------------------------------------------------------ effects


class EffectOutcome:
    """Result of claiming an irreversible effect."""

    def __init__(self, result: str, replayed: bool) -> None:
        self.result = result
        # True when the ledger already held this key, i.e. a resume re-reached a
        # step that had already committed. The caller must NOT act again.
        self.replayed = replayed


def commit_email(
    conn: sqlite3.Connection,
    idempotency_key: str,
    run_id: str,
    step: int,
    args: dict[str, Any],
    result: str,
) -> EffectOutcome:
    """Ledger the send and deliver it, atomically. Safe to call repeatedly."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT result FROM effects WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if existing is not None:
            conn.execute("COMMIT")
            return EffectOutcome(existing["result"], replayed=True)

        now = time.time()
        conn.execute(
            "INSERT INTO effects "
            "(idempotency_key, run_id, step, tool, args_json, result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                idempotency_key,
                run_id,
                step,
                "send_email",
                json.dumps(args, sort_keys=True),
                result,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO emails "
            "(idempotency_key, run_id, to_addr, subject, body, sent_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                idempotency_key,
                run_id,
                args.get("to", ""),
                args.get("subject", ""),
                args.get("body", ""),
                now,
            ),
        )
        conn.execute("COMMIT")
        return EffectOutcome(result, replayed=False)
    except Exception:
        conn.execute("ROLLBACK")
        raise


def count_emails(conn: sqlite3.Connection, run_id: str | None = None) -> int:
    if run_id is None:
        return int(conn.execute("SELECT COUNT(*) AS n FROM emails").fetchone()["n"])
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM emails WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
    )


def list_emails(conn: sqlite3.Connection, run_id: str | None = None) -> list[sqlite3.Row]:
    if run_id is None:
        return conn.execute("SELECT * FROM emails ORDER BY id").fetchall()
    return conn.execute(
        "SELECT * FROM emails WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {config.DB_PATH}")
