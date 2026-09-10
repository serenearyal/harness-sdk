"""LoopGuard on a real Agent driven through the event loop with a mocked model."""

import pytest

from strands import Agent, tool
from strands.hooks.events import BeforeToolsEvent
from strands.vended_plugins.loop_guard import DEFAULT_NUDGE, DEFAULT_STOP, LoopGuard, batch_signature
from tests.fixtures.mocked_model_provider import MockedModelProvider


@tool
def get_sensor_series(pid_id: str, minutes: int = 20) -> str:
    """Fixed values for a sensor over the last minutes."""
    return f"{pid_id}: 0.74 V flat over {minutes} min"


@tool
def read_live(pid_id: str) -> str:
    """Ask the car for a live reading right now."""
    return f"{pid_id}: 812 rpm"


SERIES = {"pid_id": "o2_b1s1_v", "minutes": 20}


def _tool_use(name: str, input: dict, uid: str) -> dict:
    return {"toolUse": {"name": name, "input": input, "toolUseId": uid}}


def _text(message) -> str:
    return " ".join(block["text"] for block in message["content"] if "text" in block)


def _tool_result_turns(agent: Agent) -> list:
    return [m for m in agent.messages if m["role"] == "user" and any("toolResult" in b for b in m["content"])]


def test_first_repeat_nudges_and_second_repeat_ends_the_turn():
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t1")]},
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t2")]},
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t3")]},
            {"role": "assistant", "content": [{"text": "never reached"}]},
        ]
    )
    state: dict = {}
    agent = Agent(model=model, tools=[get_sensor_series], plugins=[LoopGuard()])
    result = agent("Is my O2 sensor bad?", invocation_state=state)

    assert model.index == 3, "the turn ends after the second repeat, no fourth model call"
    assert result.stop_reason == "end_turn"
    assert _text(result.message) == DEFAULT_STOP

    turns = _tool_result_turns(agent)
    assert len(turns) == 3
    assert _text(turns[0]) == "", "the first batch is not a repeat: no note"
    assert _text(turns[1]) == DEFAULT_NUDGE, "the first repeat carries the note in the tool-result turn"
    guard = LoopGuard.state(state)
    assert guard is not None
    assert guard.hits == 2 and guard.stopped


def test_a_different_batch_is_not_a_repeat():
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t1")]},
            {
                "role": "assistant",
                "content": [_tool_use("get_sensor_series", {"pid_id": "o2_b1s1_v", "minutes": 60}, "t2")],
            },
            {"role": "assistant", "content": [{"text": "The sensor is fine."}]},
        ]
    )
    state: dict = {}
    agent = Agent(model=model, tools=[get_sensor_series], plugins=[LoopGuard()])
    result = agent("Is my O2 sensor bad?", invocation_state=state)

    assert model.index == 3
    assert _text(result.message) == "The sensor is fine."
    guard = LoopGuard.state(state)
    assert guard is not None
    assert guard.hits == 0
    assert all(_text(m) == "" for m in _tool_result_turns(agent))


def test_the_same_batch_in_another_order_is_a_repeat():
    a = {
        "role": "assistant",
        "content": [
            _tool_use("get_sensor_series", {"pid_id": "a", "minutes": 20}, "1"),
            _tool_use("get_sensor_series", {"minutes": 20, "pid_id": "b"}, "2"),
        ],
    }
    b = {
        "role": "assistant",
        "content": [
            _tool_use("get_sensor_series", {"pid_id": "b", "minutes": 20}, "9"),
            _tool_use("get_sensor_series", {"pid_id": "a", "minutes": 20}, "8"),
        ],
    }
    assert batch_signature(a) == batch_signature(b)
    assert batch_signature({"role": "assistant", "content": [{"text": "no tools"}]}) == ""


def test_nudge_only_when_stop_after_is_none():
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t1")]},
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t2")]},
            {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t3")]},
            {"role": "assistant", "content": [{"text": "Done after three looks."}]},
        ]
    )
    state: dict = {}
    guard = LoopGuard(stop_after=None, nudge="({tools} again? Stop.)")
    agent = Agent(model=model, tools=[get_sensor_series], plugins=[guard])
    result = agent("Is my O2 sensor bad?", invocation_state=state)

    assert model.index == 4
    assert _text(result.message) == "Done after three looks."
    notes = [_text(m) for m in _tool_result_turns(agent)]
    assert notes == ["", "(get_sensor_series again? Stop.)", "(get_sensor_series again? Stop.)"]
    seen = LoopGuard.state(state)
    assert seen is not None
    assert seen.hits == 2 and not seen.stopped


def test_an_ignored_tool_never_counts_as_a_repeat():
    live = {"pid_id": "rpm"}
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [_tool_use("read_live", live, "t1")]},
            {"role": "assistant", "content": [_tool_use("read_live", live, "t2")]},
            {"role": "assistant", "content": [_tool_use("read_live", live, "t3")]},
            {
                "role": "assistant",
                "content": [_tool_use("read_live", live, "t4"), _tool_use("get_sensor_series", SERIES, "t5")],
            },
            {
                "role": "assistant",
                "content": [_tool_use("read_live", live, "t6"), _tool_use("get_sensor_series", SERIES, "t7")],
            },
            {"role": "assistant", "content": [{"text": "Idle is steady."}]},
        ]
    )
    state: dict = {}
    guard = LoopGuard(ignore={"read_live"})
    agent = Agent(model=model, tools=[get_sensor_series, read_live], plugins=[guard])
    result = agent("Watch my idle.", invocation_state=state)

    assert model.index == 6
    assert _text(result.message) == "Idle is steady."
    # Three identical live reads: no repeat. The mixed batch repeats only
    # through its series call, so that one counts.
    seen = LoopGuard.state(state)
    assert seen is not None
    assert seen.hits == 1
    assert [_text(m) for m in _tool_result_turns(agent)] == ["", "", "", "", DEFAULT_NUDGE]
    assert batch_signature({"role": "assistant", "content": [_tool_use("read_live", live, "x")]}, {"read_live"}) == ""


def test_a_batch_split_by_an_interrupt_counts_once():
    guard = LoopGuard()
    state: dict = {}
    message = {"role": "assistant", "content": [_tool_use("get_sensor_series", SERIES, "t1")]}
    event = BeforeToolsEvent(agent=object(), message=message, invocation_state=state)  # type: ignore[arg-type]
    guard.on_before_tools(event)
    guard.on_before_tools(event)  # the same assistant message, re-fired after an interrupt
    seen = LoopGuard.state(state)
    assert seen is not None
    assert seen.hits == 0


def test_thresholds_are_validated():
    with pytest.raises(ValueError, match="nudge_after"):
        LoopGuard(nudge_after=0)
    with pytest.raises(ValueError, match="stop_after"):
        LoopGuard(nudge_after=3, stop_after=2)
