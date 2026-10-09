"""Cocotb tests for the @tvla_test decorator."""

import json
import os
from pathlib import Path

from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

import cocotb

from scasim_tvla import tvla_test

OUT = Path(os.environ["SCASIM_TVLA_OUT"])


def note(name, obj):
    (OUT / name).write_text(json.dumps(obj))


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.load.value = 0
    dut.d.value = 0
    await ClockCycles(dut.clk, 2)


async def operate(dut, x):
    dut.load.value = 1
    dut.d.value = x
    await RisingEdge(dut.clk)
    dut.load.value = 0
    await RisingEdge(dut.clk)



@tvla_test({0: 0xA5, 1: lambda rng: rng.randrange(1, 256)}, num_tests=5, warmup=2)
async def deco_pass(dut, x, seg):
    if seg.id == 0:
        await setup(dut)
    await operate(dut, x)
    note("last.json", {"id": seg.id, "x": x, "label": seg.label})


@tvla_test({0: 0xA5, 1: 0x5A}, num_tests=5, warmup=1, test_options={"timeout_time": 5, "timeout_unit": "us"})
async def deco_fail(dut, x, seg):
    if seg.id == 0:
        await setup(dut)
    await operate(dut, x)
    assert seg.id != 3, "planned failure"
