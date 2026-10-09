"""Clock and reset helpers."""

from __future__ import annotations

from typing import Any

import cocotb
from cocotb.clock import Clock
from cocotb.handle import Immediate
from cocotb.task import Task
from cocotb.triggers import ClockCycles

__all__ = ["start_clock", "reset"]


def start_clock(clock: Any, period: float, unit: str = "step", start_high: bool = False) -> Task:
    """Start a clock on `clock` and return its task.

    The clock starts low, so every rising edge is a real transition and the first one comes half
    a period after the call. Pass `start_high=True` to start with a rising edge at once.
    """
    if not start_high:
        clock.set(Immediate(0))
    return cocotb.start_soon(Clock(clock, period, unit=unit).start(start_high=start_high))


async def reset(rst: Any, clock: Any, cycles: int = 2, active_high: bool = True) -> None:
    """Hold `rst` active for `cycles` rising edges of `clock`, then release it.

    The reset is released in the write phase after the last of those edges. The function returns
    in that phase, so a driver or a test can write in the same phase.
    """
    rst.value = 1 if active_high else 0
    await ClockCycles(clock, cycles)
    rst.value = 0 if active_high else 1
