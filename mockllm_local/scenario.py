"""Scenario definitions: the script the mock model follows, loaded from YAML.

A scenario is a flat list of turns. The server keeps one cursor per session and
advances it on *every* request, so fault turns (a 429, a connection reset) each
consume one position -- that is what makes S6's `429 -> 529 -> 200` work when
the client retries with an identical request body.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

SCENARIOS_DIR = pathlib.Path(__file__).parent / "scenarios"

# Turn kinds. Anything else in a YAML file is a load-time error.
KIND_MESSAGE = "message"        # normal 200 response
KIND_HTTP_ERROR = "http_error"  # status code + headers, no model output
KIND_RESET = "reset"            # write part of the body, then RST the socket
KIND_TRUNCATE = "truncate"      # promise a full Content-Length, deliver less
KIND_HANG = "hang"              # stall, then either respond or drop

VALID_KINDS = {KIND_MESSAGE, KIND_HTTP_ERROR, KIND_RESET, KIND_TRUNCATE, KIND_HANG}


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """One tool invocation the mock model wants the agent to perform."""

    name: str
    id: str
    input: dict[str, Any] = dataclasses.field(default_factory=dict)
    # When set, this exact string is emitted instead of a serialisation of
    # `input` -- the mechanism behind S2's malformed arguments.
    raw_arguments: str | None = None


@dataclasses.dataclass(frozen=True)
class Turn:
    kind: str = KIND_MESSAGE
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None

    # kind == http_error
    status: int = 500
    headers: dict[str, str] = dataclasses.field(default_factory=dict)

    # kind == hang
    seconds: float = 30.0
    then_drop: bool = False

    # kind == message, S8: text doubles every time this turn is served.
    grow: bool = False

    def resolved_stop_reason(self) -> str:
        if self.stop_reason:
            return self.stop_reason
        return "tool_use" if self.tool_calls else "end_turn"


@dataclasses.dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    description: str
    turns: tuple[Turn, ...]
    # S4 needs a turn that never stops arriving.
    repeat_last: bool = False

    def turn_at(self, cursor: int) -> Turn | None:
        """The turn to serve at `cursor`, or None once the script is exhausted."""
        if cursor < len(self.turns):
            return self.turns[cursor]
        if self.repeat_last and self.turns:
            return self.turns[-1]
        return None

    def overrun(self, cursor: int) -> int:
        """How many times past the end of the script we are (0 while inside it)."""
        return max(0, cursor - len(self.turns) + 1)


def _parse_tool_call(raw: dict[str, Any], scenario_id: str, index: int) -> ToolCall:
    if "name" not in raw:
        raise ValueError(f"{scenario_id} turn {index}: tool call is missing 'name'")
    return ToolCall(
        name=raw["name"],
        id=raw.get("id", f"toolu_{scenario_id.lower()}_{index:02d}"),
        input=raw.get("input") or {},
        raw_arguments=raw.get("raw_arguments"),
    )


def _parse_turn(raw: dict[str, Any], scenario_id: str, index: int) -> Turn:
    kind = raw.get("kind", KIND_MESSAGE)
    if kind not in VALID_KINDS:
        raise ValueError(
            f"{scenario_id} turn {index}: unknown kind {kind!r} "
            f"(expected one of {sorted(VALID_KINDS)})"
        )

    return Turn(
        kind=kind,
        text=raw.get("text", ""),
        tool_calls=tuple(
            _parse_tool_call(call, scenario_id, index)
            for call in raw.get("tool_calls") or []
        ),
        stop_reason=raw.get("stop_reason"),
        status=int(raw.get("status", 500)),
        headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
        seconds=float(raw.get("seconds", 30.0)),
        then_drop=bool(raw.get("then_drop", False)),
        grow=bool(raw.get("grow", False)),
    )


def load_scenario(path: pathlib.Path) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenario_id = str(data.get("id") or path.stem).upper()

    turns = tuple(
        _parse_turn(raw, scenario_id, index)
        for index, raw in enumerate(data.get("turns") or [])
    )
    if not turns:
        raise ValueError(f"{path.name}: scenario has no turns")

    return Scenario(
        id=scenario_id,
        name=str(data.get("name") or path.stem),
        description=str(data.get("description") or ""),
        turns=turns,
        repeat_last=bool(data.get("repeat_last", False)),
    )


def load_all(directory: pathlib.Path = SCENARIOS_DIR) -> dict[str, Scenario]:
    """Load every scenario YAML, keyed by upper-case id (S1, S2, ...)."""
    scenarios: dict[str, Scenario] = {}
    for path in sorted(directory.glob("*.yaml")):
        scenario = load_scenario(path)
        if scenario.id in scenarios:
            raise ValueError(f"duplicate scenario id {scenario.id} in {path.name}")
        scenarios[scenario.id] = scenario
    if not scenarios:
        raise ValueError(f"no scenario YAML files found in {directory}")
    return scenarios


if __name__ == "__main__":
    for sid, scn in load_all().items():
        kinds = ", ".join(turn.kind for turn in scn.turns)
        tail = " (repeats forever)" if scn.repeat_last else ""
        print(f"{sid:>4}  {scn.name:<26} [{kinds}]{tail}")
