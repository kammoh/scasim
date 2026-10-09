"""Variant a: cocotb per-cycle Python testbench (cocotb >= 2.1).

Python drives the clock (cocotb Clock) and awaits every rising edge.
On every edge it touches K signals: reads of probe_i and writes to aux_i, alternating.
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

    cyc = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
        cyc += 1
    dut.rst_n.value = 1

    chk = 0
    acc = 0
    for s in range(segs):
        dut.seed.value = ((s + 1) * 0x9E3779B1) & MASK
        dut.start.value = 1
        await RisingEdge(dut.clk)  # E0
        cyc += 1
        dut.start.value = 0
        for _ in range(seg_len + 1):  # E1..EL, E(L+1)
            await RisingEdge(dut.clk)
            cyc += 1
            for is_read, h in touches:
                if is_read:
                    acc ^= int(h.value)
                else:
                    h.value = acc & MASK
        assert int(dut.done.value) == 1, "done not set"
        chk = (chk * 31 + int(probe0.value)) & MASK
    print(f"BENCH cycles={cyc} chk={chk}", flush=True)
