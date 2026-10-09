"""`StreamDriver` and `ReadyDriver`: drive a valid/ready stream."""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Mapping

import cocotb
from cocotb.handle import Immediate
from cocotb.triggers import ClockCycles, Event, ReadOnly, RisingEdge

from .idle import IdlePolicy
from .interface import StreamInterface

__all__ = ["StreamDriver", "ReadyDriver"]


def _own_rng(rng: random.Random | None, interface: StreamInterface, role: str) -> random.Random:
    """The random stream of a driver: the given one, or one seeded from the cocotb seed and the name."""
    if rng is not None:
        return rng
    return random.Random(f"{cocotb.RANDOM_SEED}:{role}:{interface.name}")


class StreamDriver:
    """Send transactions on a stream, as the source of `valid` and the data.

    Timing:

    - The driver writes `valid` and the data after a rising edge, in the normal write phase. It
      holds them until the transfer is accepted.
    - A transfer is accepted at the rising edge where `valid && ready` held in the settled state
      just before that edge. The driver samples `ready` in `ReadOnly` for this. Without a `ready`
      signal, every cycle with `valid` is a transfer.
    - A transaction is done after its accepting edge. `send` returns then, in the write phase
      after that edge.
    - If the next transaction is queued and the idle policy gives no idle cycles, the driver
      writes it in the same write phase. `valid` stays high and transfers go back to back.
    - If the queue is empty, `valid` goes low. A transaction queued later starts at the next
      rising edge.

    Args:
        interface: the stream to drive.
        clock: the clock signal.
        idle: idle cycles (with `valid` low) before each transaction.
        rng: the random stream for `idle`. The default is a stream of its own, seeded from the
            cocotb seed and the interface name. The global `random` module is never used.
    """

    def __init__(
        self,
        interface: StreamInterface,
        clock: Any,
        idle: IdlePolicy | None = None,
        rng: random.Random | None = None,
    ):
        self.interface = interface
        self.clock = clock
        self.idle = idle or IdlePolicy.none()
        self.rng = _own_rng(rng, interface, "driver")
        self.sent = 0  # transactions accepted so far
        self._queue: deque[tuple[Mapping[str, Any], Event | None]] = deque()
        self._parked = False
        self._wake = Event()
        self._idle_event = Event()
        self._idle_event.set()
        interface.valid.set(Immediate(0))
        self._task = cocotb.start_soon(self._run())

    def send_nowait(self, tx: Mapping[str, Any]) -> None:
        """Queue a transaction. A dict of field values. A missing or None field keeps its value."""
        self._queue.append((tx, None))
        self._idle_event.clear()
        if self._parked:
            self._parked = False
            self._wake.set()

    async def send(self, tx: Mapping[str, Any]) -> None:
        """Queue a transaction and wait until its accepting edge has passed."""
        done = Event()
        self._queue.append((tx, done))
        self._idle_event.clear()
        if self._parked:
            self._parked = False
            self._wake.set()
        await done.wait()

    async def flush(self) -> None:
        """Wait until every queued transaction has been accepted."""
        await self._idle_event.wait()

    def close(self) -> None:
        """Stop the driver."""
        self._task.cancel()

    async def _run(self) -> None:
        iface = self.interface
        valid = iface.valid
        ready = iface.ready
        write = iface.write
        clock = self.clock
        queue = self._queue
        idle = self.idle
        no_idle = idle.is_none
        draw = idle.draw
        rng = self.rng
        edge = RisingEdge(clock)
        read_only = ReadOnly()
        valid_high = False
        await edge  # start in a write phase
        while True:
            if not queue:
                if valid_high:
                    valid.value = 0
                    valid_high = False
                self._idle_event.set()
                self._parked = True
                self._wake.clear()
                await self._wake.wait()
                await edge  # a late transaction starts at the next write phase
            tx, done = queue.popleft()
            if not no_idle:
                n = draw(rng)
                if n:
                    if valid_high:
                        valid.value = 0
                        valid_high = False
                    if n == 1:
                        await edge
                    else:
                        await ClockCycles(clock, n)
            write(tx)
            if not valid_high:
                valid.value = 1
                valid_high = True
            if ready is None:
                await edge
            else:
                while True:
                    await read_only
                    try:
                        accepted = bool(ready.value)
                    except ValueError:  # x or z on ready: not ready
                        accepted = False
                    await edge
                    if accepted:
                        break
            self.sent += 1
            if done is not None:
                done.set()


class ReadyDriver:
    """Drive `ready` of a stream, as the sink.

    `backpressure` gives the number of cycles with `ready` low before each cycle with `ready`
    high. With `IdlePolicy.none()`, `ready` is always high.
    """

    def __init__(
        self,
        interface: StreamInterface,
        clock: Any,
        backpressure: IdlePolicy | None = None,
        rng: random.Random | None = None,
    ):
        if interface.ready is None:
            raise ValueError(f"stream '{interface.name}' has no ready signal to drive")
        self.interface = interface
        self.clock = clock
        self.backpressure = backpressure or IdlePolicy.none()
        self.rng = _own_rng(rng, interface, "ready")
        if self.backpressure.is_none:
            interface.ready.set(Immediate(1))
            self._task = None
        else:
            interface.ready.set(Immediate(0))
            self._task = cocotb.start_soon(self._run())

    def close(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        ready = self.interface.ready
        clock = self.clock
        draw = self.backpressure.draw
        rng = self.rng
        edge = RisingEdge(clock)
        ready_high = False
        await edge  # start in a write phase
        while True:
            n = draw(rng)
            if n:
                if ready_high:
                    ready.value = 0
                    ready_high = False
                if n == 1:
                    await edge
                else:
                    await ClockCycles(clock, n)
            if not ready_high:
                ready.value = 1
                ready_high = True
            await edge
