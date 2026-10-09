# Simulation cost model (bench/sim)

This directory measures what a Verilator simulation costs and fits a model. It
does not optimize anything. The design is synthetic, so the model does not depend
on one real design.

All builds, the cocotb venv, and the outputs go to a scratch directory, never to
the repo.

## Files

| File | Purpose |
| --- | --- |
| `gen_design.py` | Writes the SystemVerilog: design, wrapper, and pure-SV testbench. |
| `tb_percycle.py`, `tb_wrapper.py` | cocotb test modules (variants a and b). |
| `cocotb_driver.py` | Builds or runs one cocotb variant with the cocotb 2.x runner API. |
| `run_sweep.py` | Builds, runs, and measures all variants. Writes a TSV. |
| `fit_model.py` | Fits the cost model to the TSV and prints the analysis. |
| `requirements.txt` | The cocotb pin (2.1 or later; no fallback to older versions). |

## Design

`gen_design.py --regs N --width W --subs S` makes N registers of W bits in S leaf
modules under `bench_top`. W must be one of 4, 8, 12, 16, 24, 32, 48, 64, 128.
Each register is an XNOR LFSR. It advances only when its enable is set. The enable
is an 8-bit slice of a per-leaf LFSR that is below `thresh`. So the toggle density
is rho = thresh / 256 (`thresh` is a runtime input, 0 to 256).

One segment: pulse `start` for one clock (the registers load a value from `seed`),
run `seg_len` clocks, then `done` goes high. `seed` is an input, so each segment
differs. `bench_top` also has 16 `probe_i` outputs and 16 `aux_i` inputs that
cocotb touches.

## Variants

| Variant | Testbench | Models |
| --- | --- | --- |
| a | cocotb, Python drives the clock (`Clock`) and awaits every edge. It touches K signals per cycle (reads of `probe_i`, writes to `aux_i`). | The current testbench style. |
| b | cocotb with `bench_wrap` (SV): the wrapper drives clock, reset, and the segment control. Python writes the seed and a request counter, then awaits `seg_done`. | cocotb with no per-cycle Python. |
| c | Pure SV testbench `bench_tb` (`verilator --binary --timing`). Plusargs `+L= +SEGS= +THRESH= +trace`. | The floor: no cocotb. |

Each variant runs with three trace modes:

- `off`: built without tracing.
- `fst`: `--trace-fst` with the flags of `cocotb/run_tvla.py`, all signals.
- `fst_top`: the same, with `--trace-depth 1`. Only the top module is traced
  (about 50 signals). Verilator's `--trace-depth` cannot select the DUT scope in
  variants b and c: the wrapper and the DUT hierarchy count as one level. So this
  mode is a near-zero-signal trace, not a DUT-only trace.

All variants build with the Verilator flags of `run_tvla.py` (`-O3`, `--x-assign fast`,
`--x-initial fast`, `-march=native`). Variants a and b also get what the cocotb
runner adds (`--vpi --public-flat-rw`). Variant c does not. So the b-to-c
difference includes the cost of `--public-flat-rw`, not only of cocotb.

Cycle counts differ by a few per mille between variants (reset cycles, and one
idle cycle per segment in variant b). Every run prints `BENCH cycles=... chk=...`.
`run_sweep.py` uses the printed cycle count and warns if the checksum of two
variants differs for the same design and seeds.

## cocotb 2.1 environment

```bash
SCRATCH=/private/tmp/claude-501/-Volumes-src-scasim/1af4613f-4461-4614-a737-78d38e5abe6b/scratchpad
python3 -m venv $SCRATCH/venv-cocotb
$SCRATCH/venv-cocotb/bin/pip install -r bench/sim/requirements.txt
```

`cocotb_driver.py` exits if the installed cocotb is older than 2.1. cocotb 2.1
includes `cocotb_tools.runner`. Variant c needs no cocotb.

## Run the sweep and fit the model

```bash
cd bench/sim
python3 run_sweep.py --fresh                # default: 168 points, 2 repeats, 36 builds; about 15 minutes of CPU (9 for runs, 6 for builds)
python3 run_sweep.py --preset tiny --fresh  # smoke test, a few minutes
python3 run_sweep.py --preset large --fresh # more sizes, rho, K, and cycles
python3 fit_model.py $SCRATCH/simbench/sweep_default.tsv
```

Options: `--scratch DIR`, `--out TSV`, `--variants abc`, `--traces off,fst,fst_top`,
`--sizes N:W:S,...`, `--thresh T,...`, `--ks K,...`, `--segs n,...`, `--seg-len L`,
`--repeats R`. The machine may be loaded, so the TSV holds CPU time (user + system
of the simulator process tree) and the minimum over the repeats. `cpu_all_s` lists
all repeats. Wall time is not used. Builds are kept under `<scratch>/simbench/build`.

TSV columns: `variant trace N W S rho L segs K cycles nsig user_s sys_s cpu_s
cpu_all_s rss_mb fst_mb reps build_cpu_s build_wall_s`. `nsig` is an estimate of
the trace handles (4 N + 40, fitted from VCD dumps of three sizes).

## Model

`fit_model.py` fits each (variant, trace) group by non-negative weighted least
squares (weights 1/y):

```
cpu = c0 + cyc * (c1 + c2*N + c3*N*W + c4*N*rho + c5*N*rho*W/2 + c6*K)
```

`c2` and `c3` are the design and per-signal cost per cycle, `c4` and `c5` the cost
of trace changes (traced groups only), and `c6` the cost of one Python touch
(variant a). The script prints the coefficients, R-squared, and the median and
maximum relative error, then which part dominates, then the mapping of the real
setup (1,064 signals, 1.48 M time points, 150 s per batch; options `--real-*`).
