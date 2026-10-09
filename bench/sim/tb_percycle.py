"""Variant a: cocotb per-cycle Python testbench (cocotb >= 2.1).

Python drives the clock (cocotb Clock) and awaits every rising edge.
On every edge of a segment it touches K signals: reads of probe_i and writes to aux_i,
alternating. The writes always store 0, which is the value that variants b and c tie
aux_i to. So the design sees the same inputs, and the trace records the same activity,
in all variants. Only the Python cost differs.

Schedule per segment (the same in variants b and c): E0 (start), E1..EL (run),
E(L+1) (done is set; sample probe_0), then one idle edge before the next start.
"""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

NPROBE = 16
MASK = 0xFFFFFFFF


@cocotb.test()
async def per_cycle(dut):
    seg_len = int(os.environ["BENCH_L"])
    segs = int(os.environ["BENCH_SEGS"])
    thresh = int(os.environ["BENCH_THRESH"])
    k = int(os.environ["BENCH_K"])

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.seed.value = 0
    dut.seg_len.value = seg_len
    dut.thresh.value = thresh
    for i in range(NPROBE):
        getattr(dut, f"aux_{i}").value = 0

    reads = [getattr(dut, f"probe_{i}") for i in range(NPROBE)]
    writes = [getattr(dut, f"aux_{i}") for i in range(NPROBE)]
    # touch j: even j reads probe_(j/2), odd j writes aux_(j/2)
    touches = [(j % 2 == 0, (reads if j % 2 == 0 else writes)[(j // 2) % NPROBE]) for j in range(k)]
    probe0 = reads[0]

    async def tick():
        await RisingEdge(dut.clk)
        for is_read, h in touches:
            if is_read:
                nonlocal_acc[0] ^= int(h.value)
            else:
                h.value = 0

    nonlocal_acc = [0]
    cyc = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
        cyc += 1
    dut.rst_n.value = 1

    chk = 0
    for s in range(segs):
        if s > 0:
            await tick()  # idle edge between segments
            cyc += 1
        dut.seed.value = ((s + 1) * 0x9E3779B1) & MASK
        dut.start.value = 1
        await tick()  # E0
        cyc += 1
        dut.start.value = 0
        for _ in range(seg_len + 1):  # E1..EL, E(L+1)
            await tick()
            cyc += 1
        assert int(dut.done.value) == 1, "done not set"
        chk = (chk * 31 + int(probe0.value)) & MASK
    print(f"BENCH cycles={cyc} chk={chk}", flush=True)
