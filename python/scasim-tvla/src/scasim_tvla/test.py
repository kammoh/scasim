"""The `@tvla_test` decorator."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import cocotb

from .session import Tvla


def tvla_test(
    classes: Mapping[int, Any],
    num_tests: int | None = None,
    *,
    warmup: int = 1,
    test_options: Mapping[str, Any] | None = None,
    **session_options: Any,
) -> Callable:
    """Make a cocotb test from `async def op(dut, x, seg)`.

    The test builds a `Tvla` session, runs `op` once per segment with `x = seg.input`, and
    writes the metadata in a `finally` block: committed if the test passed, else diagnostic.
    `op` must return after the edge that follows its last operation edge (see `session`).

    `session_options` go to `Tvla` (for example `weights`, `schedule`, `clock`, `design_random`).
    `test_options` go to `cocotb.test` (for example `timeout_time`).
    """

    def decorate(op: Callable) -> Any:
        async def test(dut):
            tvla = Tvla(dut, classes, warmup=warmup, **session_options)
            with tvla:
                async for seg in tvla.segments(num_tests):
                    await op(dut, seg.input, seg)

        # cocotb finds the test by name and module. Do not use functools.wraps: it would make
        # cocotb see the signature of `op`.
        test.__name__ = op.__name__
        test.__qualname__ = op.__qualname__
        test.__doc__ = op.__doc__
        test.__module__ = op.__module__
        return cocotb.test(**(test_options or {}))(test)

    return decorate
