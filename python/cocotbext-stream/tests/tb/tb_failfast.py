"""With the default Scoreboard (fail_fast=True) a mismatch must fail the cocotb test.

These tests are expected to FAIL in cocotb. tests/test_stream_sim.py checks the failure text.
"""

import cocotb

from cocotbext.stream import ReadyDriver, Scoreboard, StreamDriver, StreamInterface, StreamMonitor, reset, start_clock


async def setup(dut):
    start_clock(dut.clk, 10)
    await reset(dut.rst, dut.clk, cycles=3)
    i = StreamInterface(dut, "c_in", fields=["data", "last"])
    o = StreamInterface(dut, "c_out", fields=["data", "last"])
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    return drv, StreamMonitor(o, dut.clk)


@cocotb.test()
async def fail_fast_mismatch(dut):
    drv, mon = await setup(dut)
    board = Scoreboard()
    board.add_interface(mon, "out").expect({"data": 0x1235, "last": 1})
    drv.send_nowait({"data": 0x1234, "last": 1})
    # The failure must come from the monitor task, not from this timeout.
    await board.wait_for_completion(timeout=100_000, unit="step")


@cocotb.test()
async def fail_fast_unexpected(dut):
    drv, mon = await setup(dut)
    board = Scoreboard()
    board.add_interface(mon, "out")
    drv.send_nowait({"data": 7, "last": 0})
    await board.wait_for_completion(timeout=100_000, unit="step")  # nothing is pending, so it returns
    await cocotb.triggers.ClockCycles(dut.clk, 20)
