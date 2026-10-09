"""`Scoreboard`: compare expected and observed transactions per interface."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from cocotb.triggers import Event, SimTimeoutError, with_timeout

__all__ = ["Scoreboard", "Channel", "Mismatch", "FieldDiff"]

_MISSING = object()
_LIST_LIMIT = 10  # most items printed per section of a report


@dataclass(frozen=True)
class FieldDiff:
    """One field that differs. `observed` is None when the field is absent from the observed item."""

    field: str
    expected: Any
    observed: Any
    absent: bool = False


@dataclass(frozen=True)
class Mismatch:
    """An observed item that does not match its expected item."""

    channel: str
    index: int  # position among the observed items of the channel (0-based)
    expected: Mapping[str, Any]
    observed: Mapping[str, Any]
    diffs: tuple[FieldDiff, ...]

    def __str__(self) -> str:
        lines = [f"'{self.channel}' item {self.index}: {len(self.diffs)} field(s) differ"]
        for d in self.diffs:
            if d.absent:
                lines.append(f"  {d.field}: expected {_fmt(d.expected)}, but the field is not in the observed item")
            else:
                lines.append(f"  {d.field}: expected {_fmt(d.expected)}, got {_fmt(d.observed)}")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return hex(value) if value > 9 or value < 0 else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _fmt_tx(tx: Mapping[str, Any]) -> str:
    return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in tx.items()) + "}"


def _same(expected: Any, observed: Any) -> bool:
    if observed == expected:
        return True
    if isinstance(expected, (list, tuple, str)) or isinstance(observed, (str, list)):
        return False
    try:
        return int(expected) == observed
    except (TypeError, ValueError):
        return False


def _diff(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> list[FieldDiff]:
    diffs: list[FieldDiff] = []
    for field, exp in expected.items():
        if exp is None:
            continue  # don't care
        obs = observed.get(field, _MISSING)
        if obs is _MISSING:
            diffs.append(FieldDiff(field, exp, None, absent=True))
        elif obs == exp:
            continue
        elif isinstance(exp, (list, tuple)):
            if not isinstance(obs, (list, tuple)) or len(obs) != len(exp):
                diffs.append(FieldDiff(field, list(exp), obs))
                continue
            for j, (e, o) in enumerate(zip(exp, obs)):
                if e is not None and not _same(e, o):
                    diffs.append(FieldDiff(f"{field}[{j}]", e, o))
        elif not _same(exp, obs):
            diffs.append(FieldDiff(field, exp, obs))
    return diffs


class Channel:
    """The expected and observed items of one interface. Made by `Scoreboard.add_channel`."""

    def __init__(self, board: Scoreboard, name: str):
        self.board = board
        self.name = name
        self.matched = 0
        self.observed = 0
        self._queue: deque[Mapping[str, Any]] = deque()  # in-order mode
        self._by_key: dict[Any, deque[Mapping[str, Any]]] = {}  # out-of-order mode
        self._pending = 0

    @property
    def pending(self) -> int:
        """Number of expected items that have not been observed."""
        return self._pending

    def pending_items(self) -> list[Mapping[str, Any]]:
        if self.board.in_order:
            return list(self._queue)
        return [tx for items in self._by_key.values() for tx in items]

    def expect(self, tx: Mapping[str, Any]) -> None:
        """Add one expected item. A None value in `tx` means the field is not compared."""
        if self.board.in_order:
            self._queue.append(tx)
        else:
            self._by_key.setdefault(self.board.key(tx), deque()).append(tx)
        self._pending += 1
        self.board._pending += 1
        self.board._done.clear()

    def expect_many(self, txs) -> None:
        for tx in txs:
            self.expect(tx)

    def observe(self, tx: Mapping[str, Any]) -> None:
        """Compare one observed item. This is the callback to give to a `StreamMonitor`."""
        board = self.board
        index = self.observed
        self.observed += 1
        expected = None
        if board.in_order:
            if self._queue:
                expected = self._queue.popleft()
        else:
            key = board.key(tx)
            items = self._by_key.get(key)
            if items:
                expected = items.popleft()
                if not items:
                    del self._by_key[key]
        if expected is None:
            board.unexpected.append((self.name, index, tx))
            if board.fail_fast:
                raise AssertionError(f"'{self.name}' item {index}: unexpected item {_fmt_tx(tx)}")
            return
        self._pending -= 1
        board._pending -= 1
        if board._pending == 0:
            board._done.set()
        diffs = _diff(expected, tx)
        if not diffs:
            self.matched += 1
            return
        mismatch = Mismatch(self.name, index, expected, tx, tuple(diffs))
        board.mismatches.append(mismatch)
        if board.fail_fast:
            raise AssertionError(str(mismatch))


class Scoreboard:
    """Compare expected and observed transactions, per interface.

    Args:
        in_order: if True, an observed item matches the oldest expected item of its channel.
            If False, it matches the oldest expected item with the same `key(tx)`.
        key: the key function for out-of-order mode. It gets an expected or an observed item.
        fail_fast: if True, a mismatch or an unexpected item raises `AssertionError` at once
            (in the monitor callback, so it fails the test). If False, they are collected and
            reported by `check` and `wait_for_completion`.
    """

    def __init__(
        self,
        in_order: bool = True,
        key: Callable[[Mapping[str, Any]], Any] | None = None,
        fail_fast: bool = True,
    ):
        if not in_order and key is None:
            raise ValueError("out-of-order matching needs a key function")
        self.in_order = in_order
        self.key = key
        self.fail_fast = fail_fast
        self.channels: list[Channel] = []
        self.mismatches: list[Mismatch] = []
        self.unexpected: list[tuple[str, int, Mapping[str, Any]]] = []
        self._pending = 0
        self._done = Event()
        self._done.set()

    def add_channel(self, name: str) -> Channel:
        channel = Channel(self, name)
        self.channels.append(channel)
        return channel

    def add_interface(self, monitor, name: str | None = None) -> Channel:
        """Add a channel and connect it to the monitor."""
        channel = self.add_channel(name or monitor.interface.name)
        monitor.add_callback(channel.observe)
        return channel

    @property
    def pending(self) -> int:
        """Number of expected items, over all channels, that have not been observed."""
        return self._pending

    @property
    def missing(self) -> list[tuple[str, Mapping[str, Any]]]:
        """The expected items that have not been observed, as (channel name, item)."""
        return [(ch.name, tx) for ch in self.channels for tx in ch.pending_items()]

    def report(self) -> str:
        """A text report of all mismatches, unexpected items, and missing items."""
        parts: list[str] = []
        if self.mismatches:
            parts.append(f"{len(self.mismatches)} mismatch(es):")
            parts += [str(m) for m in self.mismatches[:_LIST_LIMIT]]
            if len(self.mismatches) > _LIST_LIMIT:
                parts.append(f"... and {len(self.mismatches) - _LIST_LIMIT} more")
        if self.unexpected:
            parts.append(f"{len(self.unexpected)} unexpected item(s) (not expected, or duplicates):")
            parts += [f"  '{n}' item {i}: {_fmt_tx(tx)}" for n, i, tx in self.unexpected[:_LIST_LIMIT]]
            if len(self.unexpected) > _LIST_LIMIT:
                parts.append(f"... and {len(self.unexpected) - _LIST_LIMIT} more")
        missing = self.missing
        if missing:
            parts.append(f"{len(missing)} missing item(s) (expected, never observed):")
            parts += [f"  '{n}': {_fmt_tx(tx)}" for n, tx in missing[:_LIST_LIMIT]]
            if len(missing) > _LIST_LIMIT:
                parts.append(f"... and {len(missing) - _LIST_LIMIT} more")
        return "\n".join(parts)

    def check(self) -> None:
        """Raise `AssertionError` with the report if there is any mismatch, unexpected, or missing item."""
        text = self.report()
        if text:
            raise AssertionError("scoreboard failed:\n" + text)

    async def wait_for_completion(self, timeout: float | None = None, unit: str = "step") -> None:
        """Wait until every expected item has been observed, then call `check`.

        With `timeout`, stop waiting after that much simulation time and call `check`, which then
        reports the missing items. Items that arrive after this call returns are not seen here. To
        catch late duplicates, wait a few more cycles and call `check` again.
        """
        if self._pending:
            if timeout is None:
                await self._done.wait()
            else:
                try:
                    await with_timeout(self._done.wait(), timeout, unit)
                except SimTimeoutError:
                    pass
        self.check()
