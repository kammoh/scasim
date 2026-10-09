"""Variant b: cocotb with an HDL wrapper (cocotb >= 2.1).

bench_wrap drives clock, reset, and segment control. Python writes the seed and
a request counter, then awaits seg_done. It does not await clock edges.
"""
import os

import cocotb
from cocotb.triggers import RisingEdge

MASK = 0xFFFFFFFF


@cocotb.test()
async def wrapper(dut):
    seg_len = int(os.environ["BENCH_L"])
    segs = int(os.environ["BENCH_SEGS"])
    thresh = int(os.environ["BENCH_THRESH"])

    dut.seg_len.value = seg_len
    dut.thresh.value = thresh
    dut.seed_in.value = 0
    dut.req.value = 0
    for s in range(segs):
        dut.seed_in.value = ((s + 1) * 0x9E3779B1) & MASK
        dut.req.value = s + 1
        await RisingEdge(dut.seg_done)
    print(f"BENCH cycles={int(dut.cyc.value)} chk={int(dut.chk.value)}", flush=True)
