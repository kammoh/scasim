# cocotbext-stream

Low-overhead valid/ready stream drivers, monitors, and scoreboards for cocotb 2.1 and later.
It replaces `cocotb-bus` based helpers such as the `cocotb_ext` package in krystals_hw.

Install: `pip install -e .` (needs `cocotb>=2.1` and Python 3.10 or later). Tests: `pytest`.
The simulator tests need Verilator on `PATH`. They are skipped if Verilator or cocotb 2.1 is missing.

## Timing model

- **Driver.** It writes `valid` and the data after a rising edge, in the normal write phase. It
  holds them while `ready` is low.
- **Acceptance.** A transfer happens at the rising edge where `valid && ready` held in the settled
  state just before that edge. The driver and the monitor both read that state in `ReadOnly`.
- **Done.** A transaction is done after its accepting edge. `await driver.send(tx)` returns then.
- **Monitor.** It never writes. In each cycle it reads `valid` (and `ready`) in `ReadOnly`. It
  reads and converts the data only when a transfer happens.
- **No `ready` signal.** Every cycle with `valid` high is a transfer.

## API

```python
from cocotbext.stream import (
    StreamInterface, StreamDriver, ReadyDriver, StreamMonitor,
    Scoreboard, IdlePolicy, start_clock, reset,
)
```

- `StreamInterface(dut, prefix, separator="_", data_prefix=None, fields=None, valid=None, ready="auto")`
  resolves all handles once. `fields` is a list of names, or a `{field: hdl_name}` map.
  A name such as `"vec[0:3]"` is a flattened array (signals `vec_0` to `vec_3`, list value).
  A packed value is a wide signal and has an int value.
- `StreamDriver(iface, clock, idle=IdlePolicy.none(), rng=None)` with `send_nowait(tx)`,
  `await send(tx)`, and `await flush()`.
- `ReadyDriver(iface, clock, backpressure=IdlePolicy.none(), rng=None)` drives `ready` of a sink.
- `IdlePolicy.none()`, `.fixed(n)`, `.random(p, max_idle=None)`. Each driver has its own random stream.
- `StreamMonitor(iface, clock, callback=None)`. Transactions are dicts of ints (lists for arrays).
- `Scoreboard(in_order=True, key=None, fail_fast=True)` with `add_interface(monitor, name)`,
  `channel.expect(tx)`, `await wait_for_completion(timeout, unit)`, and `check()`.
- `start_clock(clk, period, unit="step")` and `reset(rst, clock, cycles=2, active_high=True)`.

## Port example from `cocotb_ext`

Before (krystals_hw `cocotb_ext`, `cocotb-bus`):

```python
tb = ValidReadyTB(dut, "clock", "reset")
await tb.start()
io_cmd = tb.add_driver("io_cmd", data_prefix="bits",
                       elements=["inst", "srcA", "srcB", "dst"], stalls=stall_fn(3, 2))
io_out = tb.add_monitor("io_out", data_prefix="bits",
                        elements=[f"data_{i}" for i in range(8)] + ["last"],
                        back_pressure=stall_fn(3, 3))
expected_out = []
scoreboard = BusScoreboard(dut)
scoreboard.add_interface(io_out, expected_out, strict_type=False)
io_cmd.enqueue({"inst": 0, "dst": 3})
expected_out.append({"data_0": 1, "last": True})
await scoreboard.wait_for_completion()
```

After:

```python
start_clock(dut.clock, 10)
await reset(dut.reset, dut.clock, cycles=2)

cmd = StreamInterface(dut, "io_cmd", data_prefix="bits", fields=["inst", "srcA", "srcB", "dst"])
out = StreamInterface(dut, "io_out", data_prefix="bits",
                      fields=["data[0:7]", "last"])         # data is now a list of 8 values
io_cmd = StreamDriver(cmd, dut.clock, idle=IdlePolicy.random(0.3, max_idle=2))
ReadyDriver(out, dut.clock, backpressure=IdlePolicy.random(0.3, max_idle=3))
monitor = StreamMonitor(out, dut.clock)
scoreboard = Scoreboard()
expected_out = scoreboard.add_interface(monitor, "io_out")

io_cmd.send_nowait({"inst": 0, "dst": 3})                    # was enqueue
expected_out.expect({"data": [1, None, None, None, None, None, None, None], "last": 1})
await scoreboard.wait_for_completion(timeout=1_000_000, unit="step")
```

Differences from `cocotb_ext`:

- The monitor is passive. Back pressure is a separate `ReadyDriver`.
- The idle count is a policy with its own random stream, not a callable that uses the global one.
- A flattened array is one field with a list value. `None` in an expected item (or in a list
  element) means "do not compare".
- The monitor reads in `ReadOnly`, the settled state before the edge, and not right after the
  edge wake-up. This is a timing change, not only a port.
- `Scoreboard` collects field-level mismatches, unexpected (duplicate) items, and missing items,
  and `wait_for_completion(timeout=...)` reports the missing items instead of hanging.

## Benchmark

`bench/bench_stream.py` sends the same traffic through this package and through `cocotb_ext`
(Verilator, one simulator process). It prints CPU time (`time.process_time`) per transaction and
per cycle, and checks that every run delivers exactly the sent transactions.

```
python bench/bench_stream.py --out OUT_DIR -n 20000 --reps 7 --cocotb-ext DIR
```

`DIR` holds a copy of `cocotb_ext` ported to cocotb 2.1 (needs `cocotb-bus`). The copy is not part
of this repository. Without `--cocotb-ext`, only this package runs.
