"""Driver benchmark: cocotbext-stream against krystals_hw `cocotb_ext`.

Both run the same traffic through the FIFO in tests/hdl/stream_fifo.sv, in Verilator, in one
simulator process. The script builds the design, runs bench_tb.py, and prints the CPU time per
transaction and per cycle (median of the repetitions, `time.process_time`, not wall time).

Usage:
    python bench_stream.py --out OUT_DIR [-n 20000] [--reps 5] [--cocotb-ext DIR]

`DIR` is the directory that contains the `cocotb_ext` package, ported to cocotb 2.1. Do not
commit the copy. Without it, only cocotbext-stream runs. `cocotb-bus` must be installed for
the `cocotb_ext` variants.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
HDL = HERE.parent / "tests" / "hdl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory (build, logs, results)")
    ap.add_argument("-n", type=int, default=20000, help="transactions per run")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per variant")
    ap.add_argument("--cocotb-ext", type=Path, help="directory that contains the ported cocotb_ext package")
    args = ap.parse_args()

    from cocotb_tools.runner import get_runner

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    results.unlink(missing_ok=True)

    sys.path.insert(0, str(HERE))  # the runner passes sys.path to the simulator
    if args.cocotb_ext:
        sys.path.insert(0, str(args.cocotb_ext.resolve()))
    os.environ.update(
        BENCH_N=str(args.n),
        BENCH_REPS=str(args.reps),
        BENCH_OUT=str(results),
        BENCH_WITH_OLD="1" if args.cocotb_ext else "0",
    )

    runner = get_runner("verilator")
    runner.build(
        sources=[HDL / n for n in ("stream_fifo.sv", "stream_pipe.sv", "stream_reorder.sv", "stream_top.sv")],
        hdl_toplevel="stream_top",
        build_dir=out / "build",
        build_args=["-Wno-fatal"],
        timescale=("1ns", "1ps"),
        log_file=out / "build.log",
    )
    try:
        runner.test(
            test_module="bench_tb",
            hdl_toplevel="stream_top",
            build_dir=out / "build",
            test_dir=out / "run",
            results_xml=out / "run" / "results.xml",
            seed=1,
            log_file=out / "sim.log",
        )
    except SystemExit as exc:
        sys.exit(f"simulation failed (exit {exc.code}); see {out / 'sim.log'}")

    failed = [c.get("name") for c in ET.parse(out / "run" / "results.xml").getroot().iter("testcase")
              if c.find("failure") is not None or c.find("error") is not None]
    if failed:
        sys.exit(f"cocotb tests failed: {failed}; see {out / 'sim.log'}")

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    print(report(rows, args.n))


def median(rows, variant, scenario, key):
    return statistics.median(r[key] for r in rows if r["variant"] == variant and r["scenario"] == scenario)


def report(rows, n: int) -> str:
    empties = [r for r in rows if r["variant"] == "empty"]
    empty_per_cycle = statistics.median(r["cpu_ns"] / r["cycles"] for r in empties)
    lines = [
        f"N = {n} transactions per run, {len({r['rep'] for r in rows})} repetitions, median CPU time (process_time).",
        f"Empty clock (no driver, no monitor): {empty_per_cycle / 1000:.2f} us CPU per cycle.",
        "",
        "| scenario | variant | cycles | CPU us/tx | CPU us/cycle | net us/cycle | net us/tx |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("b2b", "stall"):
        for variant in ("old", "new"):
            if not any(r["variant"] == variant and r["scenario"] == scenario for r in rows):
                continue
            cpu = median(rows, variant, scenario, "cpu_ns")
            cycles = median(rows, variant, scenario, "cycles")
            net = cpu - empty_per_cycle * cycles
            name = "cocotb_ext" if variant == "old" else "cocotbext-stream"
            lines.append(
                f"| {scenario} | {name} | {cycles:.0f} | {cpu / n / 1000:.2f} | {cpu / cycles / 1000:.2f} "
                f"| {net / cycles / 1000:.2f} | {net / n / 1000:.2f} |"
            )
    lines += [
        "",
        "net = CPU minus the empty-clock cost of the same number of cycles (simulator and clock).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
