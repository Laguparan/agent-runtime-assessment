"""R4: the trust boundary, expressed as a capability set frozen before the run.

The structural claim this file makes is narrow and testable:

    Every privilege the agent will ever hold is computed at t=0 from the
    operator's own arguments, recorded in the event log, and never re-read
    from anything the model says or any tool returns.

That is the difference between a guard and a filter. A filter scans tool output
for "ignore previous instructions" and loses to the first paraphrase. A frozen
capability set cannot be widened by content at all, because nothing downstream
of t=0 is an input to it. The model in S7 does exactly what the injection asks;
the runtime refuses anyway, and the refusal does not depend on recognising the
injection.

What this does NOT protect against is the operator granting a capability and the
model then misusing it *within* the grant -- emailing an allowed recipient the
wrong contents, for instance. That is a real hole and it is named in DECISIONS.md.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import config


class PolicyDenied(Exception):
    """A capability check failed. The message goes back to the model verbatim."""


@dataclasses.dataclass(frozen=True)
class RunPolicy:
    """Immutable for the life of a run. `frozen=True` is load-bearing, not style."""

    allow_write: bool = True
    allow_python: bool = True
    # send_email is off unless the operator names recipients. There is no
    # "allow all recipients" setting, deliberately.
    email_recipients: frozenset[str] = frozenset()
    http_hosts: frozenset[str] = config.HTTP_ALLOWED_HOSTS
    workspace: str = config.WORKSPACE_DIR

    # ------------------------------------------------------------------ checks

    def check_email(self, to: str) -> None:
        if not self.email_recipients:
            raise PolicyDenied(
                "send_email is not enabled for this run. No recipients were "
                "authorised when the run started, and that cannot be changed "
                "from inside the conversation. Report what you wanted to send "
                "instead of sending it."
            )
        recipient = (to or "").strip().lower()
        if recipient not in self.email_recipients:
            allowed = ", ".join(sorted(self.email_recipients))
            raise PolicyDenied(
                f"send_email refused: {to!r} is not an authorised recipient for "
                f"this run. Authorised recipients are: {allowed}."
            )

    def check_write(self) -> None:
        if not self.allow_write:
            raise PolicyDenied("write_file is disabled for this run.")

    def check_python(self) -> None:
        if not self.allow_python:
            raise PolicyDenied("run_python is disabled for this run.")

    def check_http_host(self, host: str | None) -> None:
        if not host:
            raise PolicyDenied("http_get refused: the URL has no host.")
        if host.lower() not in self.http_hosts:
            allowed = ", ".join(sorted(self.http_hosts))
            raise PolicyDenied(
                f"http_get refused: {host!r} is not on the allow-list. "
                f"Allowed hosts are: {allowed}. This list is fixed for the run."
            )

    # ------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_write": self.allow_write,
            "allow_python": self.allow_python,
            "email_recipients": sorted(self.email_recipients),
            "http_hosts": sorted(self.http_hosts),
            "workspace": self.workspace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunPolicy":
        return cls(
            allow_write=bool(data.get("allow_write", True)),
            allow_python=bool(data.get("allow_python", True)),
            email_recipients=frozenset(
                addr.strip().lower() for addr in data.get("email_recipients") or []
            ),
            http_hosts=frozenset(
                host.strip().lower()
                for host in data.get("http_hosts") or config.HTTP_ALLOWED_HOSTS
            ),
            workspace=data.get("workspace") or config.WORKSPACE_DIR,
        )

    @classmethod
    def from_args(cls, args: Any) -> "RunPolicy":
        """Build from parsed CLI arguments. The only place a policy is born."""
        return cls(
            allow_write=not getattr(args, "no_write", False),
            allow_python=not getattr(args, "no_python", False),
            email_recipients=frozenset(
                addr.strip().lower()
                for addr in (getattr(args, "allow_email", None) or [])
                if addr.strip()
            ),
            http_hosts=frozenset(
                host.strip().lower()
                for host in (getattr(args, "allow_host", None) or config.HTTP_ALLOWED_HOSTS)
            ),
        )
