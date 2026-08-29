"""Local stand-in for the assessment's `mockllm/` package.

The candidate brief describes `mockllm/` as *given* and says not to modify it.
That package was never received, so this is a hand-built replacement used only
to develop and test the agent runtime locally. It is deliberately kept under a
different name so it can never be mistaken for the provided one.

See mockllm_local/README.md for the fidelity notes and known divergences.
"""

__all__ = ["tokenizer", "scenario", "wire", "server"]
