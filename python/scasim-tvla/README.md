# scasim-tvla

Turns a cocotb 2.1 testbench into a TVLA leakage-assessment harness for scasim.

- `scasim_tvla.meta`: writes `scasim_meta` version 1 files. It uses only the standard library.
- `scasim_tvla.session`: the `Tvla` session. It schedules the classes, opens and closes
  segments, and writes the metadata. It needs `cocotb>=2.1`.
- `scasim_tvla.test`: the `@tvla_test` decorator.
- `scasim_tvla.runner` and the `scasim-tvla` command: build once, simulate the batches, analyze
  them with `tvla`, and merge the results. It needs only the standard library. The simulator
  work runs in child processes.

Install for development:

    pip install -e 'python/scasim-tvla[cocotb,test]'
    pytest python/scasim-tvla

The `tvla` binary is built from the scasim repository (`cargo build --release --bin tvla`). The
runner does not build it. It looks for it in `--tvla PATH`, then `SCASIM_TVLA_BIN`, then `PATH`.

## Use

    from scasim_tvla import Tvla, tvla_test

    @tvla_test({0: FIXED_INPUT, 1: gen_random}, num_tests=1000, warmup=1)
    async def my_test(dut, x, seg):
        await run_one_operation(dut, x)

or, with the session directly:

    tvla = Tvla(dut, {0: FIXED_INPUT, 1: gen_random}, warmup=1)
    with tvla:                                    # writes the metadata, also on failure
        async for seg in tvla.segments(1000):
            await run_one_operation(dut, seg.input)

A class value is a constant, or a function. A function is called with `seg.rng` if it has a
parameter named `rng` or a required positional parameter. Otherwise it is called without
an argument, and the global `random` module is seeded once per batch from the stimulus stream.

## Segment boundaries

A segment is the half-open range `[start, end)` in simulator steps. It opens when the loop body
starts and closes when the body returns. After `await RisingEdge(clk)`, the time is exactly the
time of that edge, and the activity of that edge is recorded at the same time. So a segment that
returns right after its last operation edge does not contain that edge.

**Return after the edge that follows the last operation edge.** Or end the segment with
`seg.end_at(time)`, or `await seg.extend(1, clk)` as the last statement of the body.

## Schedules and random streams

- `schedule="iid"` (default): each label is drawn independently with the class `weights`.
- `schedule="blocks"`: shuffled blocks. The block size is the smallest integer count that has the
  ratio of the weights. The remainder at the end is drawn iid. Blocks can correlate a label with the
  previous state, and Welch's test assumes independence.
- Warm-up segments run first. They use their own random stream and consume no schedule draws.
  Their ids are skipped in the file.
- The schedule, stimulus, idle, design, and warm-up streams are seeded by SHA-256 of
  (base seed, batch id, stream name). All derived seeds are in the metadata.

## Design randomness

`Tvla(..., design_random=hook, design_random_mode="on" | "off")`. The hook is called as
`hook(design_rng, mode)` before the warm-up. It may be `async`. It returns the applied mode
as a string, or a dict `{"applied": ..., "how": ...}`. A mode without a hook is an error. When no
mode is requested, `batch.design_random` is `{"requested": "none", "applied": "none", "how": "none"}`.

## Environment variables

The runner sets them. The arguments of `Tvla` override them.

| Variable | Meaning |
|---|---|
| `SCASIM_TVLA_SEED` | base seed (default: `cocotb.RANDOM_SEED`) |
| `SCASIM_TVLA_BATCH` | batch id (default `b0000`) |
| `SCASIM_TVLA_OUT` | output directory; the file is `meta.json` there (default: the current directory) |
| `SCASIM_TVLA_TESTS` | number of segments; overrides `num_tests` |
| `SCASIM_TVLA_WAVEFORM` | waveform file name, relative to the output directory |
| `SCASIM_TVLA_DESIGN_RANDOM` | `on` or `off` |

## Failure behavior

The metadata is written synchronously (never with `await`), in a `finally` block or in
`with tvla:`. It is `committed` if the test passed, else `diagnostic`. An open segment is
never listed. If the loop ends early (`break`), the file is a `diagnostic` with
`extensions.diagnostic.reason` = "schedule not completed: k of n segments"; a shortened run is never committed. Do not `await` in your own `finally` blocks: cocotb raises a second error when the
test times out.

## The runner

    scasim-tvla run --sources design.f --toplevel dut_top \
        --test-module keccak.sca.KeccakCore_tb --testcase sha3_256_test \
        --batches 20 --tests-per-batch 5000 --out OUT --jobs 4 \
        -- --clock dut_top.clock --include scope:dut_top --exclude signal:dut_top.clock -d 2

Run it from the directory that holds the test module (the current directory goes on the module
path; `--pythonpath DIR` adds more). The arguments after `--` go to `tvla`.

- **Build.** One Verilator build with `-O3 --trace-fst` and your `--build-arg` values (for
  example `--build-arg=-GLEAK=0`). The build is reused while the sources, the arguments, the trace
  scope rules, and the Verilator and cocotb versions stay the same. `--sources` takes files and file
  lists (`.f` or `.list`; a line that starts with `-` or `+` is a Verilator argument).
- **Trace volume.** `--trace-scope SCOPE` writes a `.vlt` file that turns tracing off for the whole
  design and on for `SCOPE`. `--trace-off-rule SCOPE` turns it off again for a part. Use Verilator scope
  names without a `TOP.` prefix. `--trace-depth` is passed on, but it does not select the DUT scope
  reliably.
- **Signal check.** After the build, the runner simulates one short probe batch and runs
  `tvla --list-signals` with your rules. It stops if nothing is selected, if a rule matches no
  signal, or if `--clock` names no signal of the waveform. Version 1 metadata needs `--clock`.
- **Batches.** Each batch has its own `Runner`, directory `OUT/bNNNN`, seed, and trace file, and a
  clean environment: `COCOTB_*` and `SCASIM_TVLA_*` from your shell are removed. The runner reads
  `results.xml` and the metadata, not the exit code. The seed of a batch comes from `--seed` and
  the batch id, so a retry or a different `--jobs` gives the same data.
- **Pipeline.** A committed batch goes to `tvla --stats-out`, which writes `statistics.bin`. Then the
  waveform is deleted (`--keep none`, the default). `--keep waveform` keeps it. `--keep traces` is
  not supported: `tvla` cannot write per-batch traces for metadata version 1.
  Simulation pauses when `--jobs` finished waveforms wait for analysis, so the disk holds at most
  `2 x jobs` waveforms. The default `--jobs` comes from the cores, the RAM, and the free disk.
- **Failures.** A failed batch keeps its waveform and a `diagnostic.txt`. It is left out of the merge.
  The command exits with 1 after it merges the other batches.
- **Reruns.** `OUT/manifest.json` records the state of each batch (`simulated`, `cached`, `failed`).
  A rerun skips cached batches, analyzes batches that wait with a waveform, and simulates failed
  batches again with the same seed. If the sources, the test, the number of tests per batch, or the
  preprocessing options change, all batches run again.
- **Merge.** `tvla --merge-stats` merges the caches in batch-id order into `OUT/report`.
  `--curve every|every:K|final` is passed on. `tvla` rejects preprocessing options in a merge, so the
  runner passes them only to the per-batch calls.
- **`--no-analyze`** only simulates and keeps the waveforms.
- **`--profile`** sets `COCOTB_ENABLE_PROFILING=1` for the simulation and records the user and
  system CPU time of each simulator in the manifest and in `report/run.json`. It also writes
  `profile.txt` (the top of the Python profile) in each batch directory. Without `--profile`, nothing
  is set or imported for it.

For a Makefile flow that runs the simulations itself, point the testbench at a batch directory
(`SCASIM_TVLA_OUT`) and then run:

    scasim-tvla collect OUT -- --clock dut_top.clock --include scope:dut_top
    scasim-tvla merge OUT --tvla path/to/tvla

`collect` finds the batch directories (`OUT/*/meta.json`), writes `meta.list` and the manifest.
`merge` analyzes the batches that still wait with a waveform, then merges.

## Migration from a krystals_hw cocotb 1.9 test

The test below is the shape of `keccak/sca/KeccakCore_tb.py`, summarized.

**Before** (cocotb 1.9, `cocotb_ext`, `run_tvla.py`):

```python
from cocotb.utils import get_sim_time
from cocotb_ext import BusScoreboard, ValidReadyTB

@cocotb.test()
async def sha3_256_test(dut):
    num_tests = int(os.environ.get("NUM_TESTS", 100))
    tb = ValidReadyTB(dut, "clock", reset="reset", reset_value=1)
    await tb.start(clock_period=10)
    io_in = tb.add_driver("io_din", elements=[...], data_prefix=None)
    io_out = tb.add_monitor("io_dout", elements=[...], data_prefix=None)
    scoreboard = BusScoreboard(dut, custom_comparator=compare_fn)
    scoreboard.add_interface(io_out, expected_out, strict_type=False)
    meta_markers = []
    for _ in range(num_tests):
        meta_class = random.randint(0, 1)
        meta_start_time = get_sim_time()
        io_in.append(shared_input(fixed_input if meta_class == 0 else gen_rand_input()))
        expected_out.extend(expected_words)
        assert await scoreboard.wait_for_completion()
        meta_markers.append((meta_start_time, get_sim_time(), meta_class))
    save_meta(meta_markers[1:], clock_period, [])      # gzip JSON, the first marker is dropped
```

**After** (cocotb 2.1, `cocotbext-stream`, `scasim_tvla`):

```python
from cocotb.triggers import RisingEdge
from cocotbext.stream import (StreamDriver, StreamInterface, StreamMonitor, reset,
                              start_clock)
from scasim_tvla import Tvla

@cocotb.test()
async def sha3_256_test(dut):
    start_clock(dut.clock, 10)
    await reset(dut.reset, dut.clock, cycles=2)
    io_in = StreamDriver(StreamInterface(dut, "io_din", fields=[...]), dut.clock)
    io_out = StreamMonitor(StreamInterface(dut, "io_dout", fields=[...]), dut.clock)
    expected = []
    tvla = Tvla(dut, {0: fixed_input, 1: gen_rand_input}, warmup=1, clock=dut.clock)
    with tvla:                                      # writes meta.json, also when the test fails
        async for seg in tvla.segments():           # the number of tests comes from the runner
            io_in.send_nowait(shared_input(seg.input))
            expected.extend(expected_words(seg.input))
            while len(io_out.transactions) < len(expected):   # wait for the output words
                await RisingEdge(dut.clock)
            await seg.extend(1, dut.clock)          # see "Segment boundaries"
        # after the loop: unshare the monitor's transactions and compare them with `expected`
```

What changed:

- **Metadata.** `save_meta`, the marker list, the first-marker drop, and the gzip file are gone.
  `Tvla` writes `meta.json` (scasim_meta version 1) with the segments, the seeds, and the status
  (`committed` or `diagnostic`). The first segment is the warm-up (`warmup=1`). There is no
  `clock_period`: pass `--clock PATH` to `tvla` (after `--`), and the samples follow the clock edges.
- **Classes.** `random.randint(0, 1)` becomes the session schedule (`schedule="iid"` by default,
  with `weights=` to change the odds). A class value is a constant or a function. A function that takes
  `rng` gets the stimulus stream of the session. Without `rng`, the global `random` module is seeded
  once per batch from that stream. The seeds are in the metadata.
- **Simulation side.** `NUM_TESTS`, `TRACE_FILENAME`, and the `TVLA` switch are replaced by
  `scasim-tvla run --tests-per-batch ... ` (it sets `SCASIM_TVLA_TESTS`, `SCASIM_TVLA_WAVEFORM`, and the
  other `SCASIM_TVLA_*` variables). `run_tvla.py` and joblib are not needed. The runner also does the
  analysis, so there is no separate `tvla --meta-list` step.
- **Boundary.** A segment is `[start, end)`. Return after the edge that follows the last operation
  edge, or end with `await seg.extend(1, clk)`.

### cocotb 1.9 to 2.1

| cocotb 1.9 | cocotb 2.1 |
|---|---|
| `from cocotb.utils import get_sim_time` | `from cocotb.simtime import get_sim_time` (the old name still works) |
| `handle.setimmediatevalue(v)` | `handle.set(Immediate(v))` with `from cocotb.handle import Immediate` |
| `Clock(sig, period, units="ns")` | `Clock(sig, period, unit="ns")` |
| `await cocotb.start(task)` | `cocotb.start_soon(task)`; add `await Timer(0)` if the next line needs the task to run |
| `from cocotb.runner import get_runner` | `from cocotb_tools.runner import get_runner` |
| `Runner.test()` exit code | read `results.xml` with `get_results()`; it can exit with 0 after a failed test, or raise `SystemExit(1)` |
| `TOPLEVEL`, `MODULE` variables | `COCOTB_TOPLEVEL`, `COCOTB_TEST_MODULES`; a value in your shell overrides the arguments of `Runner.test()` |

### `cocotb_ext` to `cocotbext-stream`

| `cocotb_ext` (cocotb-bus) | `cocotbext.stream` |
|---|---|
| `ValidReadyTB(dut, "clock", reset="reset")`, `tb.start()` | `start_clock(dut.clock, 10)`, `await reset(dut.reset, dut.clock)` |
| `tb.add_driver(name, elements=..., data_prefix=..., stalls=fn)` | `StreamDriver(StreamInterface(dut, name, fields=..., data_prefix=...), clock, idle=IdlePolicy.random(p, max_idle=n))` |
| `driver.append(tx)` | `driver.send_nowait(tx)`, or `await driver.send(tx)` to wait for the accepting edge |
| `tb.add_monitor(name, ..., back_pressure=fn)` | `StreamMonitor(StreamInterface(...), clock)` and `ReadyDriver(iface, clock, backpressure=IdlePolicy...)` |
| `BusScoreboard`, `add_interface(mon, expected_list)` | `Scoreboard()`, `channel = board.add_interface(monitor, name)`, `channel.expect(tx)` |
| `await scoreboard.wait_for_completion()` | `await board.wait_for_completion(timeout=..., unit="step")`; it reports the missing items |
| `custom_comparator=fn` | no equivalent. Expected items are field dictionaries (`None` means do not compare). To compare an unshared value, compute it from the monitor's transactions, as `tests/e2e/tb_leaky.py` does. `Scoreboard(in_order=False, key=fn)` matches out of order |
| `add_rand_driver` (random data on a valid-only port) | no equivalent: use a `StreamDriver` with `tvla.design_rng`, or `Tvla(design_random=hook)` |

`cocotbext-stream` has more detail in its own README (timing model, the monitor reads in
`ReadOnly`, and the idle policies each have their own random stream).
