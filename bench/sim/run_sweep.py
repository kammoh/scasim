#!/usr/bin/env python3
"""Sweep the simulation cost of the three testbench variants (stdlib only).

Builds each variant with Verilator, runs it, and records CPU time (user+sys),
max RSS, FST size, and build time. All build and output files go to --scratch.
Output: one TSV row per point (minimum CPU time over the repeats).

Variants:   a = cocotb per-cycle Python, b = cocotb + HDL wrapper, c = no cocotb
Trace modes: off | fst (all signals) | fst_top (--trace-depth 1: only the top module, about 50 signals)
"""
import argparse
import itertools
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # keep the repo clean
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_design  # noqa: E402

DEFAULT_SCRATCH = ("/private/tmp/claude-501/-Volumes-src-scasim/"
                   "1af4613f-4461-4614-a737-78d38e5abe6b/scratchpad")

# Same Verilator flags as cocotb/run_tvla.py (build_simulation).
BASE_FLAGS = ["-Wno-fatal", "-Wno-lint", "-Wno-style", "-Wno-UNOPTFLAT", "-O3",
              "--x-assign", "fast", "--x-initial", "fast", "-j", "0",
              "-CFLAGS", "-march=native -mtune=native"]
TRACE_FLAGS = ["--trace-fst", "--trace-underscore", "--trace-structs",
               "--trace-max-array", "16384", "--trace-max-width", "16384",
               "--trace-threads", "2"]

TOP = {"a": "bench_top", "b": "bench_wrap", "c": "bench_tb"}
SRC = {"a": ["bench_dut.sv"], "b": ["bench_dut.sv", "bench_wrap.sv"], "c": ["bench_dut.sv", "bench_tb.sv"]}
TOP_DEPTH = {"a": 1, "b": 1, "c": 1}  # --trace-depth 1: top module only (near-zero signals)

COLUMNS = ["variant", "trace", "N", "W", "S", "rho", "L", "segs", "K", "cycles", "nsig",
           "user_s", "sys_s", "cpu_s", "cpu_all_s", "rss_mb", "fst_mb", "reps",
           "build_cpu_s", "build_wall_s"]

PRESETS = {
    # sizes (N:W:S), thresholds (rho*256), segment counts (L = 1000), K values for variant a
    "tiny": dict(sizes=["16:16:2", "64:32:4"], thresh=[128], segs=[20], ks=[0, 4], segs_extra=[]),
    "default": dict(sizes=["16:16:2", "64:32:4", "256:32:8", "512:64:16"],
                    thresh=[26, 128, 230], segs=[100], ks=[0, 16], segs_extra=[300]),
    "large": dict(sizes=["16:16:2", "64:32:4", "256:32:8", "512:64:16", "1024:64:32", "2048:128:32"],
                  thresh=[13, 26, 77, 128, 192, 230, 256], segs=[100], ks=[0, 4, 16, 64], segs_extra=[300, 1000]),
}


def mb(rss):
    return rss / (1 << 20) if sys.platform == "darwin" else rss / 1024


def timed(cmd, cwd, log):
    """Run cmd; return (rusage dict of the child tree, returncode). Output goes to the file log."""
    t0 = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(p.pid, 0)
    return ({"user": ru.ru_utime, "sys": ru.ru_stime, "maxrss": ru.ru_maxrss,
             "wall": time.monotonic() - t0}, os.waitstatus_to_exitcode(status))


def driver(py, args, log):
    """Run cocotb_driver.py; its last stdout line is the JSON rusage of its children."""
    with open(log, "w") as f:
        rc = subprocess.call([py, os.path.join(HERE, "cocotb_driver.py")] + args, stdout=f, stderr=subprocess.STDOUT)
    lines = open(log).read().strip().splitlines()
    try:
        return json.loads(lines[-1]), rc
    except (ValueError, IndexError):
        return None, rc or 1


class Sweep:
    def __init__(self, a):
        self.a = a
        self.scratch = os.path.join(a.scratch, "simbench")
        self.py = os.path.join(a.scratch, "venv-cocotb", "bin", "python")
        self.builds = {}
        os.makedirs(self.scratch, exist_ok=True)

    def trace_args(self, v, trace):
        if trace == "off":
            return []
        args = list(TRACE_FLAGS)
        if trace == "fst_top":
            args += ["--trace-depth", str(TOP_DEPTH[v])]
        return args

    def build(self, v, trace, n, w, s):
        key = (v, trace, n, w, s)
        if key in self.builds:
            return self.builds[key]
        name = f"{v}_{trace}_{n}_{w}_{s}"
        bdir = os.path.join(self.scratch, "build", name)
        sdir = os.path.join(self.scratch, "src", f"{n}_{w}_{s}")
        shutil.rmtree(bdir, ignore_errors=True)
        os.makedirs(bdir)
        gen_design.write_design(n, w, s, sdir)
        srcs = [os.path.join(sdir, f) for f in SRC[v]]
        flags = BASE_FLAGS + self.trace_args(v, trace)
        log = os.path.join(bdir, "build.log")
        if v == "c":
            cmd = (["verilator", "--binary", "--timing", "--top-module", TOP[v], "-Mdir", os.path.join(bdir, "obj"),
                    "--timescale", "1ns/1ps"] + flags + srcs)
            ru, rc = timed(cmd, bdir, log)
        else:
            extra = ["--timing"] if v == "b" else []
            args = ["build", "--top", TOP[v], "--build-dir", os.path.join(bdir, "obj"), "--src"] + srcs
            args += ["--args-json", json.dumps(flags + extra)]
            if trace != "off":
                args += ["--waves"]
            ru, rc = driver(self.py, args, log)
        if rc != 0 or ru is None:
            raise RuntimeError(f"build failed ({name}); see {log}")
        res = (os.path.join(bdir, "obj"), ru["user"] + ru["sys"], ru["wall"])
        self.builds[key] = res
        return res

    def run_once(self, v, trace, bdir, thresh, L, segs, K, tag):
        rdir = os.path.join(self.scratch, "run", tag)
        shutil.rmtree(rdir, ignore_errors=True)
        os.makedirs(rdir)
        log = os.path.join(rdir, "run.log")
        fst = os.path.join(rdir, "trace.fst")
        if v == "c":
            cmd = [os.path.join(bdir, "Vbench_tb"), f"+L={L}", f"+SEGS={segs}", f"+THRESH={thresh}"]
            if trace != "off":
                cmd.append("+trace")
            ru, rc = timed(cmd, rdir, log)
        else:
            env = [f"BENCH_L={L}", f"BENCH_SEGS={segs}", f"BENCH_THRESH={thresh}", f"BENCH_K={K}"]
            args = ["run", "--top", TOP[v], "--build-dir", bdir,
                    "--module", "tb_percycle" if v == "a" else "tb_wrapper", "--env"] + env
            if trace != "off":
                args += ["--waves", "--trace-file", fst]
            ru, rc = driver(self.py, args, log)
        text = open(log).read()
        m = re.search(r"BENCH cycles=(\d+) chk=(\d+)", text)
        if rc != 0 or ru is None or not m:
            raise RuntimeError(f"run failed ({tag}); see {log}")
        size = os.path.getsize(fst) / 1e6 if os.path.exists(fst) else 0.0
        shutil.rmtree(rdir, ignore_errors=True)
        return dict(cycles=int(m.group(1)), chk=m.group(2), user=ru["user"], sys=ru["sys"],
                    rss=mb(ru["maxrss"]), fst=size)

    def points(self):
        p = PRESETS[self.a.preset]
        a = self.a
        sizes = a.sizes.split(",") if a.sizes else p["sizes"]
        thresh = [int(x) for x in a.thresh.split(",")] if a.thresh else p["thresh"]
        ks = [int(x) for x in a.ks.split(",")] if a.ks else p["ks"]
        segs = [int(x) for x in a.segs.split(",")] if a.segs else p["segs"]
        extra = p["segs_extra"]
        mid = 128
        for size in sizes:
            n, w, s = (int(x) for x in size.split(":"))
            for v in a.variants:
                for trace in a.traces:
                    pts = [(t, sg, 4 if v == "a" else 0) for t in thresh for sg in segs]
                    pts += [(mid, sg, 4 if v == "a" else 0) for sg in extra]
                    if v == "a":
                        pts += [(mid, segs[0], k) for k in ks]
                    seen = set()
                    for pt in pts:
                        if pt not in seen:
                            seen.add(pt)
                            yield v, trace, n, w, s, pt[0], a.seg_len, pt[1], pt[2]

    def run(self):
        a = self.a
        new = not os.path.exists(a.out) or a.fresh
        out = open(a.out, "w" if new else "a")
        if new:
            out.write("\t".join(COLUMNS) + "\n")
        checks = {}
        for v, trace, n, w, s, th, L, segs, K in self.points():
            bdir, bcpu, bwall = self.build(v, trace, n, w, s)
            tag = f"{v}_{trace}_{n}_{w}_{s}_{th}_{L}_{segs}_{K}"
            reps = [self.run_once(v, trace, bdir, th, L, segs, K, tag) for _ in range(a.repeats)]
            best = min(reps, key=lambda r: r["user"] + r["sys"])
            # Same design and seeds must give the same checksum in every variant and trace mode.
            ck = checks.setdefault((n, w, s, th, L, segs), best["chk"])
            if ck != best["chk"]:
                print(f"WARNING: checksum differs for {tag}: {ck} vs {best['chk']}", file=sys.stderr)
            row = dict(variant=v, trace=trace, N=n, W=w, S=s, rho=th / 256, L=L, segs=segs, K=K,
                       cycles=best["cycles"], nsig=gen_design.n_signals(n, w, s),
                       user_s=best["user"], sys_s=best["sys"], cpu_s=best["user"] + best["sys"],
                       cpu_all_s=",".join(f"{r['user'] + r['sys']:.3f}" for r in reps),
                       rss_mb=best["rss"], fst_mb=best["fst"], reps=len(reps),
                       build_cpu_s=bcpu, build_wall_s=bwall)
            out.write("\t".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in COLUMNS) + "\n")
            out.flush()
            print(f"{tag}: cpu {row['cpu_s']:.3f}s cycles {row['cycles']} fst {row['fst_mb']:.1f}MB", flush=True)
        out.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=DEFAULT_SCRATCH, help="scratch dir (builds, venv-cocotb, outputs)")
    ap.add_argument("--out", help="TSV output (default: <scratch>/simbench/sweep_<preset>.tsv)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
    ap.add_argument("--variants", default="abc", help="subset of 'abc'")
    ap.add_argument("--traces", default="off,fst,fst_top", help="comma list of off,fst,fst_top")
    ap.add_argument("--sizes", help="N:W:S comma list (overrides the preset)")
    ap.add_argument("--thresh", help="enable thresholds 0..256, rho = t/256 (comma list)")
    ap.add_argument("--ks", help="K values for variant a (comma list)")
    ap.add_argument("--segs", help="segment counts (comma list)")
    ap.add_argument("--seg-len", type=int, default=1000, help="L: cycles per segment")
    ap.add_argument("--repeats", type=int, default=2, help="runs per point; the minimum CPU time is kept")
    ap.add_argument("--fresh", action="store_true", help="overwrite the TSV instead of appending")
    a = ap.parse_args()
    a.traces = a.traces.split(",")
    a.variants = list(a.variants)
    if not a.out:
        a.out = os.path.join(a.scratch, "simbench", f"sweep_{a.preset}.tsv")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if not os.path.exists(os.path.join(a.scratch, "venv-cocotb", "bin", "python")) and any(v in "ab" for v in a.variants):
        sys.exit("cocotb venv not found: see README (python3 -m venv <scratch>/venv-cocotb; pip install -r requirements.txt)")
    Sweep(a).run()


if __name__ == "__main__":
    main()
