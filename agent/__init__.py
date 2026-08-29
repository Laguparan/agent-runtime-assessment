"""Adversarial agent runtime (Part A).

Module map:

    config.py    limits and budgets, in one place
    policy.py    R4 -- the capability set, frozen at t=0
    paths.py     workspace confinement
    tools.py     the five tools, argument validation, sandboxing
    storage.py   R2 -- append-only event log and the exactly-once effect ledger
    trace.py     R6 -- fsynced JSONL traces
    tokens.py    tokenizer resolution
    memory.py    R3 -- anchored-window compaction
    client.py    resilient transport (S5, S6, S12)
    loop.py      R1/R5 -- the loop, its budgets, and its stop reasons
    replay.py    R6 -- decision replay with no server
    cli.py       run / resume / replay / inspect
"""
