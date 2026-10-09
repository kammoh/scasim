"""Testbench for leaky.sv: one TVLA test, written like a functional-verification test.

It uses cocotbext-stream for the input and the output, and scasim_tvla for the segments.
Environment variables (set by the end-to-end tests; the runner leaves them alone):

- TB_CYCLES: rising edges between the accepting edge and the end of the segment (default 6)
- TB_FAIL_BATCH: the batch id that fails at segment 4
"""

import os

import cocotb
from cocotb.triggers import ClockCycles
from cocotbext.stream import StreamDriver, StreamInterface, StreamMonitor, start_clock
from scasim_tvla import Tvla

FIXED = 0xFFFF
CYCLES = int(os.environ.get("TB_CYCLES", "6"))
FAIL_BATCH = os.environ.get("TB_FAIL_BATCH")


def random_input(rng):
    return rng.getrandbits(16)


@cocotb.test()
async def leak_test(dut):
    start_clock(dut.clk, 10)
    driver = StreamDriver(StreamInterface(dut, "in", fields=["data"]), dut.clk)
    monitor = StreamMonitor(StreamInterface(dut, "out", fields=["share", "mask"]), dut.clk)
    sent = []
    tvla = Tvla(dut, {0: FIXED, 1: random_input}, warmup=2, clock=dut.clk)
    with tvla:
        async for seg in tvla.segments():
            x = seg.input
            sent.append(x)
            await driver.send({"data": x})
            await seg.extend(CYCLES, dut.clk)
            if tvla.batch == FAIL_BATCH and seg.id == 4:
                raise AssertionError("planned failure")
        await ClockCycles(dut.clk, 8)  # let the last output arrive
        received = [t["share"] ^ t["mask"] for t in monitor.transactions]
        assert received == sent, "the pipeline must return the data it was given"
