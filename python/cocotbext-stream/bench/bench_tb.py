"""cocotb side of the driver benchmark. Run by bench_stream.py, not by pytest.

Each measurement is its own cocotb test, so all tasks of a variant end with the test.
The CPU time (`time.process_time_ns`) covers only the traffic part, from the first `send` to the
last accepted item. Environment: BENCH_N, BENCH_REPS, BENCH_OUT (JSON lines file),
BENCH_WITH_OLD (1 to run the cocotb_ext variants).
"""

from __future__ import annotations

import json
import os
import random
import time

import cocotb
from cocotb.simtime import get_sim_time
from cocotb.triggers import ClockCycles, Event

from cocotbext.stream import (
    IdlePolicy,
    ReadyDriver,
    StreamDriver,
    StreamInterface,
    StreamMonitor,
    reset,
    start_clock,
)

N = int(os.environ.get("BENCH_N", "20000"))
REPS = int(os.environ.get("BENCH_REPS", "5"))
OUT = os.environ["BENCH_OUT"]
WITH_OLD = os.environ.get("BENCH_WITH_OLD") == "1"
PERIOD = 10
FIELDS = ["data", "last", "vec_0", "vec_1"]


def make_txs(n):
    rng = random.Random(42)
    return [{f: rng.getrandbits(16 if f == "data" else 8 if f.startswith("vec") else 1) for f in FIELDS} for _ in range(n)]


def record(variant, scenario, rep, cpu_ns, cycles, txs_done):
    with open(OUT, "a") as fh:
        fh.write(
            json.dumps(
                {"variant": variant, "scenario": scenario, "rep": rep, "cpu_ns": cpu_ns, "cycles": cycles, "txs": txs_done}
            )
            + "\n"
        )


async def run_new(dut, scenario):
    start_clock(dut.clk, PERIOD)
    await reset(dut.rst, dut.clk, cycles=3)
    i = StreamInterface(dut, "c_in", fields=FIELDS)
    o = StreamInterface(dut, "c_out", fields=FIELDS)
    stalled = scenario == "stall"
    idle = IdlePolicy.fixed(1) if stalled else IdlePolicy.none()
    drv = StreamDriver(i, dut.clk, idle=idle)
    ReadyDriver(o, dut.clk, backpressure=idle)
    done = Event()
    got = []

    def on_tx(tx):
        got.append(tx)
        if len(got) == N:
            done.set()

    StreamMonitor(o, dut.clk, callback=on_tx)
    txs = make_txs(N)
    t0 = get_sim_time("step")
    c0 = time.process_time_ns()
    for tx in txs:
        drv.send_nowait(tx)
    await done.wait()
    cpu = time.process_time_ns() - c0
    cycles = (get_sim_time("step") - t0) // PERIOD
    assert got == txs, "new: output differs from input"
    return cpu, cycles


async def run_old(dut, scenario):
    from cocotb_ext import ValidReadyTB

    tb = ValidReadyTB(dut, "clk", "rst")
    await tb.start()
    stalled = scenario == "stall"
    fn = (lambda: 1) if stalled else None
    drv = tb.add_driver("c_in", elements=FIELDS, stalls=fn)
    done = Event()
    got = []

    def on_tx(tx):
        got.append(tx)
        if len(got) == N:
            done.set()

    tb.add_monitor("c_out", elements=FIELDS, back_pressure=fn, callback=on_tx)
    txs = make_txs(N)
    t0 = get_sim_time("step")
    c0 = time.process_time_ns()
    drv.extend(txs)
    await done.wait()
    cpu = time.process_time_ns() - c0
    cycles = (get_sim_time("step") - t0) // PERIOD
    assert [{k: int(v) for k, v in tx.items()} for tx in got] == txs, "old: output differs from input"
    return cpu, cycles


async def run_empty(dut, cycles):
    start_clock(dut.clk, PERIOD)
    await reset(dut.rst, dut.clk, cycles=3)
    c0 = time.process_time_ns()
    await ClockCycles(dut.clk, cycles)
    return time.process_time_ns() - c0, cycles


def register(name, fn):
    globals()[name] = cocotb.test(name=name)(fn)


def make_test(variant, scenario, rep):
    runner = {"new": run_new, "old": run_old}[variant]

    async def test(dut):
        cpu, cycles = await runner(dut, scenario)
        record(variant, scenario, rep, cpu, cycles, N)

    return test


def make_empty(cycles, rep):
    async def test(dut):
        cpu, n = await run_empty(dut, cycles)
        record("empty", f"{cycles}", rep, cpu, n, 0)

    return test


for _rep in range(REPS):
    for _scenario in ("b2b", "stall"):
        register(f"new_{_scenario}_r{_rep}", make_test("new", _scenario, _rep))
        if WITH_OLD:
            register(f"old_{_scenario}_r{_rep}", make_test("old", _scenario, _rep))
    register(f"empty_r{_rep}", make_empty(2 * N, _rep))
