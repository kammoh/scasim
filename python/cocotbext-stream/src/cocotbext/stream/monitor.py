"""`StreamMonitor`: a passive monitor of a valid/ready stream."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge

from .interface import StreamInterface

__all__ = ["StreamMonitor"]


class StreamMonitor:
    """Record the transfers of a stream.

    In every cycle the monitor waits for `ReadOnly` and looks at `valid`. The state in `ReadOnly`
    is the settled state just before the next rising edge, so `valid && ready` there means that
    the transfer happens at that edge. Only then does the monitor read the data fields and
    convert them to Python values. The monitor never writes a signal.

    Args:
        interface: the stream to watch.
        clock: the clock signal.
        callback: called with each transaction (a dict of field values) when it completes.
        record: if True, keep the transactions in `transactions`. The default is True when no
            callback is given.
    """

    def __init__(
        self,
        interface: StreamInterface,
        clock: Any,
        callback: Callable[[Mapping[str, Any]], None] | None = None,
        record: bool | None = None,
    ):
        self.interface = interface
        self.clock = clock
        self.callbacks: list[Callable[[Mapping[str, Any]], None]] = [callback] if callback else []
        self.record = (callback is None) if record is None else record
        self.transactions: list[dict[str, Any]] = []
        self.count = 0
        self._task = cocotb.start_soon(self._run())

    def add_callback(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        self.callbacks.append(callback)

    def close(self) -> None:
        self._task.cancel()

    async def _run(self) -> None:
        valid = self.interface.valid
        ready = self.interface.ready
        read = self.interface.read
        edge = RisingEdge(self.clock)
        read_only = ReadOnly()
        transactions = self.transactions
        callbacks = self.callbacks
        while True:
            await read_only
            try:
                accepted = bool(valid.value) and (ready is None or bool(ready.value))
            except ValueError:  # x or z on a control signal: not a transfer
                accepted = False
            if accepted:
                tx = read()
                self.count += 1
                if self.record:
                    transactions.append(tx)
                for callback in callbacks:
                    callback(tx)
            await edge
