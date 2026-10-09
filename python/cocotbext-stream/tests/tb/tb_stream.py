"""cocotb tests for cocotbext.stream. Run by tests/test_stream_sim.py."""

import random

import cocotb
from cocotb.simtime import get_sim_time
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge

from cocotbext.stream import (
    IdlePolicy,
    ReadyDriver,
    Scoreboard,
    StreamDriver,
    StreamInterface,
    StreamMonitor,
    reset,
    start_clock,
)

PERIOD = 10
TIMEOUT = 200_000  # steps


clock_start = 0


async def setup(dut):
    global clock_start
    clock_start = get_sim_time("step")
    start_clock(dut.clk, PERIOD)
    await reset(dut.rst, dut.clk, cycles=3)


def fifo(dut, p):
    fields = ["data", "last", "vec[0:1]"]
    return StreamInterface(dut, f"{p}_in", fields=fields), StreamInterface(dut, f"{p}_out", fields=fields)


def make_txs(n, seed=1):
    rng = random.Random(seed)
    return [
        {"data": rng.getrandbits(16), "last": rng.getrandbits(1), "vec": [rng.getrandbits(8), rng.getrandbits(8)]}
        for _ in range(n)
    ]


def is_edge_time(t):
    return (t - clock_start - PERIOD // 2) % PERIOD == 0


async def run_fifo(dut, p, txs, idle=None, backpressure=None, seed=0):
    """Send `txs` through FIFO `p` and check the output in order. Return (in monitor, out monitor)."""
    i, o = fifo(dut, p)
    drv = StreamDriver(i, dut.clk, idle=idle or IdlePolicy.none(), rng=random.Random(seed))
    ReadyDriver(o, dut.clk, backpressure=backpressure or IdlePolicy.none(), rng=random.Random(seed + 1))
    mon_i = StreamMonitor(i, dut.clk)
    mon_o = StreamMonitor(o, dut.clk)
    board = Scoreboard()
    ch = board.add_interface(mon_o, "out")
    for tx in txs:
        ch.expect(tx)
        drv.send_nowait(tx)
    await board.wait_for_completion(timeout=TIMEOUT, unit="step")
    assert mon_i.transactions == txs, "the input monitor must see exactly the accepted transfers"
    return mon_i, mon_o


@cocotb.test()
async def registered_ready(dut):
    await setup(dut)
    txs = make_txs(40)
    await run_fifo(dut, "r", txs, backpressure=IdlePolicy.fixed(2))


@cocotb.test()
async def fifo_drains_last_item_at_the_accepting_edge(dut):
    await setup(dut)
    i, o = fifo(dut, "c")
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    mon_i = StreamMonitor(i, dut.clk)
    mon_o = StreamMonitor(o, dut.clk)
    sent = make_txs(3, seed=2)
    for tx in sent:
        await drv.send(tx)
        # send() returns after the accepting edge, so the monitor has recorded the item.
        assert is_edge_time(get_sim_time("step"))
        assert mon_i.count == sent.index(tx) + 1
        await ClockCycles(dut.clk, 6)  # let the FIFO drain: its last item leaves at an accepting edge
    assert mon_o.transactions == sent
    assert mon_i.transactions == sent


@cocotb.test()
async def back_to_back(dut):
    await setup(dut)
    times = []
    txs = make_txs(8, seed=3)
    i, o = fifo(dut, "c")
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    StreamMonitor(i, dut.clk, callback=lambda tx: times.append(get_sim_time("step")))
    mon_o = StreamMonitor(o, dut.clk)
    for tx in txs:
        drv.send_nowait(tx)
    await drv.flush()
    await ClockCycles(dut.clk, 4)
    assert len(times) == 8
    assert [b - a for a, b in zip(times, times[1:])] == [PERIOD] * 7
    assert mon_o.transactions == txs


@cocotb.test()
async def stalls_hold_data_stable(dut):
    await setup(dut)
    i, o = fifo(dut, "c")
    stalls = 0
    prev = None

    async def checker():
        nonlocal stalls, prev
        while True:
            await ReadOnly()
            cur = (
                int(dut.c_in_valid.value),
                int(dut.c_in_ready.value),
                int(dut.c_in_data.value),
                int(dut.c_in_last.value),
                int(dut.c_in_vec_0.value),
                int(dut.c_in_vec_1.value),
            )
            if prev is not None and prev[0] and not prev[1]:
                assert cur[0] == 1, "valid dropped before the transfer"
                assert cur[2:] == prev[2:], "data changed while the sink stalled"
            if cur[0] and not cur[1]:
                stalls += 1
            prev = cur
            await RisingEdge(dut.clk)

    cocotb.start_soon(checker())
    txs = make_txs(24, seed=4)
    await run_fifo(dut, "c", txs, backpressure=IdlePolicy.fixed(4))
    assert stalls > 0, "the test must stall the driver"


@cocotb.test()
async def no_ready_signal(dut):
    await setup(dut)
    i = StreamInterface(dut, "p_in", fields=["data"], ready=None)
    o = StreamInterface(dut, "p_out", fields=["data"])
    assert i.ready is None and not i.has_ready
    assert o.ready is None, "auto detection finds no p_out_ready"
    drv = StreamDriver(i, dut.clk)
    mon_i = StreamMonitor(i, dut.clk)
    mon_o = StreamMonitor(o, dut.clk)
    board = Scoreboard()
    ch = board.add_interface(mon_o, "out")
    txs = [{"data": 100 + k} for k in range(10)]
    for tx in txs:
        ch.expect(tx)
        drv.send_nowait(tx)
    await board.wait_for_completion(timeout=TIMEOUT, unit="step")
    assert mon_i.transactions == txs


@cocotb.test()
async def random_idle_and_backpressure(dut):
    await setup(dut)
    n = 300
    txs = make_txs(n, seed=5)
    start = get_sim_time("step")
    await run_fifo(
        dut, "c", txs, idle=IdlePolicy.random(0.5), backpressure=IdlePolicy.random(0.5), seed=11
    )
    cycles = (get_sim_time("step") - start) // PERIOD
    assert cycles > 1.5 * n, f"idle and backpressure should slow the stream, took {cycles} cycles"


@cocotb.test()
async def explicit_signal_map(dut):
    await setup(dut)
    i = StreamInterface(
        dut, valid="c_in_valid", ready="c_in_ready", fields={"d": "c_in_data", "l": "c_in_last"}
    )
    o = StreamInterface(
        dut, valid="c_out_valid", ready="c_out_ready", fields={"d": "c_out_data", "l": "c_out_last"}
    )
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    mon = StreamMonitor(o, dut.clk)
    txs = [{"d": 0xBEEF, "l": 1}, {"d": 0x1234, "l": 0}]
    for tx in txs:
        await drv.send(tx)
    await ClockCycles(dut.clk, 6)
    assert mon.transactions == txs


@cocotb.test()
async def interface_errors(dut):
    await setup(dut)
    try:
        StreamInterface(dut, "c_in", fields=["nope"])
    except AttributeError as exc:
        assert "c_in_nope" in str(exc)
    else:
        raise AssertionError("a missing signal must raise AttributeError")
    i, _ = fifo(dut, "c")
    try:
        i.write({"bogus": 1})
    except KeyError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("an unknown field must raise KeyError")
    try:
        StreamInterface(dut, "c_in", fields=["data"], ready="c_in_missing")
    except AttributeError:
        pass
    else:
        raise AssertionError("an explicit ready name must exist")


@cocotb.test()
async def scoreboard_mismatch_report(dut):
    await setup(dut)
    i, o = fifo(dut, "c")
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    mon = StreamMonitor(o, dut.clk)
    board = Scoreboard(fail_fast=False)
    ch = board.add_interface(mon, "out")
    ch.expect({"data": 0x1235, "last": 1, "vec": [1, 3]})
    drv.send_nowait({"data": 0x1234, "last": 1, "vec": [1, 2]})
    try:
        await board.wait_for_completion(timeout=TIMEOUT, unit="step")
    except AssertionError as exc:
        text = str(exc)
    else:
        raise AssertionError("the mismatch must fail the scoreboard")
    assert "data: expected 0x1235, got 0x1234" in text, text
    assert "vec[1]: expected 3, got 2" in text, text
    assert "last" not in text, text
    assert [d.field for d in board.mismatches[0].diffs] == ["data", "vec[1]"]


@cocotb.test()
async def out_of_order_matching(dut):
    await setup(dut)
    i = StreamInterface(dut, "o_in", fields=["data"])
    o = StreamInterface(dut, "o_out", fields=["data"])
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    mon = StreamMonitor(o, dut.clk)
    ooo = Scoreboard(in_order=False, key=lambda tx: tx["data"])
    in_order = Scoreboard(fail_fast=False)
    ooo_ch = ooo.add_interface(mon, "out")
    in_order_ch = in_order.add_interface(mon, "out")
    txs = [{"data": 10 + k} for k in range(6)]  # the DUT swaps each pair
    for tx in txs:
        ooo_ch.expect(tx)
        in_order_ch.expect(tx)
        drv.send_nowait(tx)
    await ooo.wait_for_completion(timeout=TIMEOUT, unit="step")
    assert [tx["data"] for tx in mon.transactions] == [11, 10, 13, 12, 15, 14]
    assert len(in_order.mismatches) == 6, "the in-order board must see the swap"


@cocotb.test()
async def missing_item_at_completion(dut):
    await setup(dut)
    i, o = fifo(dut, "c")
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    mon = StreamMonitor(o, dut.clk)
    board = Scoreboard()
    ch = board.add_interface(mon, "out")
    txs = make_txs(3, seed=6)
    for tx in txs:
        ch.expect(tx)
    ch.expect({"data": 0xDEAD, "last": 0, "vec": [0, 0]})  # the DUT never sends this
    for tx in txs:
        drv.send_nowait(tx)
    try:
        await board.wait_for_completion(timeout=50 * PERIOD, unit="step")
    except AssertionError as exc:
        assert "missing" in str(exc) and "0xdead" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("a missing item must fail at completion")
    assert ch.matched == 3 and board.pending == 1


@cocotb.test()
async def duplicate_item_at_completion(dut):
    await setup(dut)
    i, o = fifo(dut, "c")
    drv = StreamDriver(i, dut.clk)
    ReadyDriver(o, dut.clk)
    mon = StreamMonitor(o, dut.clk)
    tx = make_txs(1, seed=7)[0]
    in_order = Scoreboard(fail_fast=False)
    keyed = Scoreboard(in_order=False, key=lambda t: t["data"], fail_fast=False)
    for board in (in_order, keyed):
        board.add_interface(mon, "out").expect(tx)
    drv.send_nowait(tx)
    drv.send_nowait(tx)  # the second copy has no expected item
    await in_order.wait_for_completion(timeout=TIMEOUT, unit="step")
    await ClockCycles(dut.clk, 10)
    for board in (in_order, keyed):
        assert len(board.unexpected) == 1
        try:
            board.check()
        except AssertionError as exc:
            assert "unexpected" in str(exc)
        else:
            raise AssertionError("a duplicate must fail the check")
