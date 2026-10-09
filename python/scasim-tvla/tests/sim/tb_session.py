"""Cocotb tests for the session. They run inside the simulator, selected by name."""

import json
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.simtime import get_sim_time
from cocotb.triggers import ClockCycles, RisingEdge, Timer

from scasim_tvla import Tvla

OUT = Path(os.environ["SCASIM_TVLA_OUT"])
PERIOD = 10000  # steps: 10 ns at 1 ps


def note(name, obj):
    (OUT / name).write_text(json.dumps(obj))


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.load.value = 0
    dut.d.value = 0
    await ClockCycles(dut.clk, 2)


async def operate(dut, x):
    """One operation: load x at the next edge, then wait one more edge (the rule in the docs)."""
    dut.load.value = 1
    dut.d.value = x
    await RisingEdge(dut.clk)  # the operation edge: q changes here
    dut.load.value = 0
    await RisingEdge(dut.clk)  # the edge after it: the segment closes here


def classes():
    counter = iter(range(1, 256))
    return {0: lambda: next(counter), 1: lambda rng: rng.randrange(1, 256)}


@cocotb.test()
async def basic(dut):
    await setup(dut)
    seen = []
    tvla = Tvla(dut, {0: 0xA5, 1: lambda rng: rng.randrange(1, 256)}, warmup=1, clock=dut.clk)
    with tvla:
        async for seg in tvla.segments(6):
            x = seg.input
            await operate(dut, x)
            seen.append({"id": seg.id, "label": seg.label, "start": seg.start,
                         "end": get_sim_time("step"), "x": x, "warmup": seg.warmup})
    note("observed.json", {"precision": cocotb.simtime.time_precision, "segments": seen})


@cocotb.test()
async def fail_with(dut):
    await setup(dut)
    tvla = Tvla(dut, {0: 0xA5, 1: 0x5A}, warmup=1)
    with tvla:
        async for seg in tvla.segments(6):
            await operate(dut, seg.input)
            if seg.id == 3:
                raise AssertionError("planned failure")


@cocotb.test()
async def fail_bare(dut):
    await setup(dut)
    tvla = Tvla(dut, {0: 0xA5, 1: 0x5A}, warmup=1)
    async for seg in tvla.segments(6):
        await operate(dut, seg.input)
        if seg.id == 3:
            raise AssertionError("planned failure")


@cocotb.test(timeout_time=300, timeout_unit="ns")
async def timeout_with(dut):
    await setup(dut)
    tvla = Tvla(dut, {0: 0xA5, 1: 0x5A}, warmup=0)
    with tvla:
        async for seg in tvla.segments(1000):
            await operate(dut, seg.input)


@cocotb.test()
async def boundary(dut):
    """Activity at the final operation edge. Mode A closes at that edge, B and C after it."""
    await setup(dut)
    seen = []
    tvla = Tvla(dut, classes(), warmup=1, clock=dut.clk, schedule="blocks")
    with tvla:
        async for seg in tvla.segments(6):
            mode = "ABC"[seg.index % 3] if not seg.warmup else "A"
            dut.load.value = 1
            dut.d.value = seg.input
            await RisingEdge(dut.clk)
            edge = get_sim_time("step")
            dut.load.value = 0
            if mode == "A":
                pass  # returns right at the operation edge: this edge is NOT in the segment
            elif mode == "B":
                await seg.extend(1)  # one more edge: the operation edge is inside
            else:
                await RisingEdge(dut.clk)
                seg.end_at(edge + 1)
            seen.append({"id": seg.id, "warmup": seg.warmup, "mode": mode, "start": seg.start,
                         "edge": edge})
    note("observed.json", {"segments": seen})

