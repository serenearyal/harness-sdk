"""Loop guard: signature every tool batch, act on a repeat."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from ...hooks.events import AfterToolsEvent, BeforeToolsEvent
from ...plugins import Plugin, hook
from ...types.content import Message

DEFAULT_NUDGE = (
    "(You already requested exactly these calls earlier and have their results "
    "above. Do not repeat them - use what you have or pursue a new line of inquiry.)"
)
DEFAULT_STOP = "I kept requesting the same data, so I am stopping here with what I have."

_STATE_KEY = "loop_guard"


@dataclass
class LoopGuardState:
    """What the guard saw during one invocation. Lives in ``invocation_state``.

    Attributes:
        seen: Signatures of every tool batch the model requested so far.
        hits: How many batches were repeats of an earlier one.
        last_repeat: The signature of the batch just flagged, until its
            AfterToolsEvent consumes it.
        last_message_id: Identity of the assistant message the last
            BeforeToolsEvent carried, so a batch split by an interrupt is
            counted once.
        stopped: True once the guard ended the turn.
    """

    seen: set[str] = field(default_factory=set)
    hits: int = 0
    last_repeat: str | None = None
    last_message_id: int | None = None
    stopped: bool = False


def batch_signature(message: Message, ignore: Collection[str] = ()) -> str:
    """Return one string per distinct tool batch: the tool names and their inputs.

    Order-insensitive, and insensitive to key order inside an input, so a
    batch the model re-issues with its calls shuffled is still the same
    batch. Tool use ids are ignored: they change on every call. Calls to a
    tool in ``ignore`` drop out, and a batch of only those has no signature.

    Args:
        message: The assistant message carrying the ``toolUse`` blocks.
        ignore: Names of tools whose calls never count.

    Returns:
        The signature, or an empty string when nothing in the message counts.
    """
    calls = [
        json.dumps(
            {"name": block["toolUse"]["name"], "input": block["toolUse"].get("input")},
            sort_keys=True,
            default=str,
        )
        for block in message.get("content", [])
        if "toolUse" in block and block["toolUse"]["name"] not in ignore
    ]
    return "\n".join(sorted(calls))


class LoopGuard(Plugin):
    """End the loop when the agent asks for the same tool batch again.

    Two thresholds, both counted in repeats of an earlier batch within one
    invocation:

    * at ``nudge_after`` repeats the guard appends ``nudge`` as a text block
      to the tool-result message, so the model reads it right under the
      results it already has;
    * at ``stop_after`` repeats the guard sets ``AfterToolsEvent.end_turn``
      to ``stop``, and the run ends with that text as the final answer.
      Pass ``stop_after=None`` to nudge only.

    State lives in ``invocation_state`` under the key ``"loop_guard"``, so it
    resets per invocation and survives interrupt and resume.
    """

    name = "loop-guard"

    def __init__(
        self,
        *,
        nudge_after: int = 1,
        stop_after: int | None = 2,
        nudge: str = DEFAULT_NUDGE,
        stop: str = DEFAULT_STOP,
        ignore: Collection[str] = (),
    ) -> None:
        """Initialize the guard.

        Args:
            nudge_after: Repeats before the note. Default 1.
            stop_after: Repeats before the turn ends. Default 2. ``None`` never stops.
            nudge: The note appended to the tool-result turn. ``{tools}`` expands to
                the repeated tool names.
            stop: The final assistant text when the guard ends the turn. ``{tools}``
                expands the same way.
            ignore: Names of tools whose calls never count: a live reading or a
                clock is asked again on purpose because its answer changes.

        Raises:
            ValueError: If ``nudge_after`` is below 1 or ``stop_after`` is below ``nudge_after``.
        """
        if nudge_after < 1:
            raise ValueError("`nudge_after` must be at least 1")
        if stop_after is not None and stop_after < nudge_after:
            raise ValueError("`stop_after` must be at least `nudge_after`")
        self.nudge_after = nudge_after
        self.stop_after = stop_after
        self.nudge = nudge
        self.stop = stop
        self.ignore = frozenset(ignore)
        super().__init__()

    @staticmethod
    def state(carrier: Any) -> LoopGuardState | None:
        """Return the guard's state for an invocation.

        Args:
            carrier: The ``invocation_state`` dict given to the agent, or an object
                carrying one (a hook event). An ``AgentResult`` does not carry it.

        Returns:
            The state, or None when the guard has not seen a tool batch yet.
        """
        if isinstance(carrier, dict):
            return carrier.get(_STATE_KEY)
        return getattr(carrier, "invocation_state", {}).get(_STATE_KEY)

    @hook
    def on_before_tools(self, event: BeforeToolsEvent) -> None:
        """Signature the requested batch and count a repeat.

        Args:
            event: The event carrying the assistant message with ``toolUse`` blocks.
        """
        state = event.invocation_state.setdefault(_STATE_KEY, LoopGuardState())
        # A per-tool interrupt re-fires this event for the same assistant
        # message; that is one batch, counted once.
        if state.last_message_id == id(event.message):
            return
        state.last_message_id = id(event.message)
        signature = batch_signature(event.message, self.ignore)
        if not signature:
            return
        if signature in state.seen:
            state.hits += 1
            state.last_repeat = signature
        else:
            state.seen.add(signature)
            state.last_repeat = None

    @hook
    def on_after_tools(self, event: AfterToolsEvent) -> None:
        """Nudge or end the turn when the batch just executed was a repeat.

        Args:
            event: The event carrying the user-role tool-result message.
        """
        state: LoopGuardState | None = event.invocation_state.get(_STATE_KEY)
        if state is None or state.last_repeat is None:
            return
        tools = ", ".join(sorted({json.loads(c)["name"] for c in state.last_repeat.split("\n")}))
        state.last_repeat = None
        if self.stop_after is not None and state.hits >= self.stop_after:
            state.stopped = True
            event.end_turn = self.stop.format(tools=tools)
        elif state.hits >= self.nudge_after:
            # The tool-result message is the same dict the loop appends to
            # history, so a text block added here is what the model reads next.
            event.message["content"].append({"text": self.nudge.format(tools=tools)})
