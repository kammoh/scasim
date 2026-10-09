#!/usr/bin/env python3
"""Build or run one cocotb variant (cocotb >= 2.1, runner API cocotb_tools.runner).

Run with the cocotb venv's Python. run_sweep.py calls this and reads the last
line of stdout (JSON): the CPU time of the child processes only (build tools or
the simulator), so the Python start-up of this driver does not count.

  cocotb_driver.py build --top T --build-dir D --src F... --args-json JSON [--waves]
  cocotb_driver.py run   --top T --build-dir D --module M [--waves] [--trace-file F]
                         [--env K=V...]
"""
import argparse
import json
import os
import resource
import sys
import time

import cocotb
from cocotb_tools.runner import Verilog, get_runner

if tuple(int(x) for x in cocotb.__version__.split(".")[:2]) < (2, 1):
    sys.exit(f"cocotb >= 2.1 required, found {cocotb.__version__}")


def children():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime, r.ru_stime, r.ru_maxrss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "run"])
    ap.add_argument("--top", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--src", nargs="*", default=[])
    ap.add_argument("--args-json", default="[]", help="JSON list of Verilator build arguments")
    ap.add_argument("--waves", action="store_true")
    ap.add_argument("--module")
    ap.add_argument("--trace-file")
    ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()

    runner = get_runner("verilator")
    t0 = time.monotonic()
    if a.mode == "build":
        runner.build(
            sources=a.src,
            hdl_toplevel=a.top,
            build_dir=a.build_dir,
            build_args=[Verilog(x) for x in json.loads(a.args_json)],
            timescale=("1ns", "1ps"),
            waves=a.waves,
            always=True,
            clean=True,
        )
    else:
        runner.test(
            test_module=a.module,
            hdl_toplevel=a.top,
            hdl_toplevel_lang="verilog",
            build_dir=a.build_dir,
            test_dir=os.path.dirname(os.path.abspath(__file__)),
            extra_env=dict(e.split("=", 1) for e in a.env),
            waves=a.waves,
            test_args=["--trace-file", a.trace_file] if a.waves else [],
            timescale=("1ns", "1ps"),
            seed=1,
        )
    wall = time.monotonic() - t0
    u, s, rss = children()
    print(json.dumps({"user": u, "sys": s, "maxrss": rss, "wall": wall}))


if __name__ == "__main__":
    main()
