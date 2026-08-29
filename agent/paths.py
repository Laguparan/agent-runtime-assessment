"""Workspace confinement.

The obvious implementation is `abspath(join(root, p)).startswith(root)`, and it
is wrong twice over: `/tmp/workspace_evil` starts with `/tmp/workspace`, and
`abspath` does not resolve symlinks, so a link inside the workspace pointing out
of it passes. This uses `realpath` on both sides and `commonpath` for the
containment test, which has neither failure.
"""

from __future__ import annotations

import os


class PathDenied(Exception):
    """Resolution left the workspace. The message goes back to the model."""


def safe_path(workspace: str, candidate: str) -> str:
    """Resolve `candidate` inside `workspace`, or raise PathDenied."""
    if not isinstance(candidate, str) or not candidate.strip():
        raise PathDenied("path must be a non-empty string relative to the workspace.")

    root = os.path.realpath(workspace)
    os.makedirs(root, exist_ok=True)

    if os.path.isabs(candidate) or (len(candidate) > 1 and candidate[1] == ":"):
        raise PathDenied(
            f"path must be relative to the workspace; {candidate!r} is absolute."
        )

    # realpath resolves symlinks, so a link inside the workspace that points out
    # of it is caught here rather than silently followed.
    resolved = os.path.realpath(os.path.join(root, candidate))

    try:
        contained = os.path.commonpath([root, resolved]) == root
    except ValueError:
        # Different drives on Windows -- commonpath refuses, which is a denial.
        contained = False

    if not contained:
        raise PathDenied(
            f"path {candidate!r} resolves outside the workspace and was refused. "
            "All file access is confined to the workspace directory."
        )
    return resolved


def relative(workspace: str, resolved: str) -> str:
    """Workspace-relative form of an already-resolved path, for logs."""
    try:
        return os.path.relpath(resolved, os.path.realpath(workspace)).replace("\\", "/")
    except ValueError:
        return resolved
