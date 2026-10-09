# scasim-tvla

Turns a cocotb 2.1 testbench into a TVLA leakage-assessment harness for scasim.

- `scasim_tvla.meta`: writes `scasim_meta` version 1 files. It uses only the standard library.
- `scasim_tvla.session`: the `Tvla` session. It schedules the classes, opens and closes
  segments, and writes the metadata. It needs `cocotb>=2.1`.
- `scasim_tvla.test`: the `@tvla_test` decorator.

Install for development:

    pip install -e 'python/scasim-tvla[cocotb,test]'
    pytest python/scasim-tvla

The command `scasim-tvla` is reserved for the runner (not implemented yet).

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
never listed. Do not `await` in your own `finally` blocks: cocotb raises a second error when the
test times out.
