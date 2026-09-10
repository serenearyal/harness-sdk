"""Loop guard plugin for Strands Agents — stop an agent that repeats a tool batch.

Example:
    ```python
    from strands import Agent
    from strands.vended_plugins.loop_guard import LoopGuard

    agent = Agent(tools=[get_sensor_series], plugins=[LoopGuard()])
    agent("Is my O2 sensor bad?")
    ```
"""

from .plugin import DEFAULT_NUDGE, DEFAULT_STOP, LoopGuard, LoopGuardState, batch_signature

__all__ = [
    "DEFAULT_NUDGE",
    "DEFAULT_STOP",
    "LoopGuard",
    "LoopGuardState",
    "batch_signature",
]
