"""The `scasim-tvla` command: `run`, `collect`, and `merge`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _args, runner


def _split(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split at the first `--`: (arguments of scasim-tvla, arguments for tvla)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scasim-tvla",
        description="Build, simulate, and analyze the batches of a TVLA run. Put the arguments "
        "of tvla after `--` (for example: -- --clock top.clk --include scope:top).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="build, simulate the batches, analyze, and merge")
    r.add_argument("--sources", nargs="+", required=True, metavar="FILE",
                   help="source files, or file lists (.f or .list) with one path per line")
    r.add_argument("--toplevel", required=True, help="the top-level module")
    r.add_argument("--test-module", required=True, help="the cocotb test module")
    r.add_argument("--testcase", help="the cocotb test to run")
    r.add_argument("--batches", type=int, required=True)
    r.add_argument("--tests-per-batch", type=int, required=True, metavar="K")
    r.add_argument("--out", type=Path, required=True, metavar="DIR")
    r.add_argument("--seed", type=int, help="base seed (default: the seed of an earlier run in "
                   "DIR, else random)")
    r.add_argument("--jobs", type=int, help="parallel simulations (default: from cores, RAM, disk)")
    r.add_argument("--build-arg", action="append", default=[], metavar="ARG",
                   help="an extra Verilator argument (repeatable; use --build-arg=-GX=1)")
    r.add_argument("--pythonpath", action="append", default=[], metavar="DIR",
                   help="a directory for the test module (repeatable; the current one is always used)")
    r.add_argument("--trace-scope", action="append", default=[], metavar="SCOPE",
                   help="trace only this scope (Verilator name, no TOP. prefix; repeatable)")
    r.add_argument("--trace-off-rule", action="append", default=[], metavar="SCOPE",
                   help="turn tracing off for this scope or pattern (repeatable)")
    r.add_argument("--trace-depth", type=int, metavar="D",
                   help="Verilator --trace-depth (it does not select the DUT scope reliably)")
    r.add_argument("--design-random", choices=("on", "off"))
    r.add_argument("--keep", nargs="+", choices=("none", "traces", "waveform"), default=["none"],
                   metavar="WHAT",
                   help="what stays per batch besides the statistics cache: none (default), "
                   "traces (the per-channel traces, channel-traces.npz), waveform, or traces and "
                   "waveform together (--keep traces waveform)")
    r.add_argument("--traces-channels", nargs="+", default=[], metavar="SPEC",
                   help="with --keep traces: write only these channels (names, or regex:PATTERN); "
                   "default all")
    r.add_argument("--no-analyze", action="store_true", help="only simulate; keep the waveforms")
    r.add_argument("--curve", metavar="every|every:K|final", help="passed to tvla --merge-stats")
    r.add_argument("--profile", action="store_true",
                   help="profile the Python side of the simulation and report the simulator CPU")
    r.add_argument("--tvla", metavar="PATH", help="the tvla binary (else SCASIM_TVLA_BIN, else PATH)")

    c = sub.add_parser("collect", help="make meta.list and the manifest from batch directories")
    c.add_argument("dir", type=Path)
    c.add_argument("--curve", metavar="every|every:K|final")
    c.add_argument("--keep", choices=("none", "waveform"), default="waveform")

    m = sub.add_parser("merge", help="analyze the batches that are not cached, merge, and report")
    m.add_argument("dir", type=Path)
    m.add_argument("--curve", metavar="every|every:K|final")
    m.add_argument("--tvla", metavar="PATH")
    return p


def main(argv: list[str] | None = None) -> int:
    own, tvla_args = _split(list(sys.argv[1:] if argv is None else argv))
    parser = _parser()
    ns = parser.parse_args(own)
    try:
        if ns.command == "run":
            return _run(ns, tvla_args, parser)
        if ns.command == "collect":
            m = runner.collect(ns.dir, tvla_args, ns.curve, ns.keep)
            states = [r.get("state") for r in m.batches().values()]
            print(f"scasim-tvla: {len(states)} batches in {ns.dir}: "
                  + ", ".join(f"{states.count(s)} {s}" for s in runner.Manifest.STATES))
            return 0
        report = runner.merge_dir(ns.dir, ns.tvla, tvla_args or None, ns.curve)
        return _finish(report)
    except (runner.RunnerError, _args.ArgsError, FileNotFoundError, ValueError) as exc:
        print(f"scasim-tvla: error: {exc}", file=sys.stderr)
        return 2


def _finish(report: runner.RunReport) -> int:
    if report.report_dir is not None:
        print(f"scasim-tvla: report in {report.report_dir}")
    for batch, reason in report.failed.items():
        print(f"scasim-tvla: {batch} failed: {reason}", file=sys.stderr)
    return report.exit_code


def _run(ns: argparse.Namespace, tvla_args: list[str], parser: argparse.ArgumentParser) -> int:
    try:
        keep = runner.normalize_keep(ns.keep)
    except runner.RunnerError as exc:
        parser.error(str(exc))
    if "traces" in runner.keep_set(keep) and ns.no_analyze:
        parser.error("--keep traces needs the analysis: do not use --no-analyze")
    if ns.traces_channels and "traces" not in runner.keep_set(keep):
        parser.error("--traces-channels needs --keep traces")
    if ns.batches < 1 or ns.tests_per_batch < 1:
        parser.error("--batches and --tests-per-batch must be at least 1")
    cfg = runner.RunConfig(
        out=ns.out, batches=ns.batches, tests_per_batch=ns.tests_per_batch, seed=ns.seed,
        jobs=ns.jobs, keep=keep, analyze=not ns.no_analyze, tvla=ns.tvla, tvla_args=tvla_args,
        curve=ns.curve, profile=ns.profile, design_random=ns.design_random,
        traces_channels=ns.traces_channels,
        sim=runner.SimSpec(
            sources=ns.sources, toplevel=ns.toplevel, test_module=ns.test_module,
            testcase=ns.testcase, build_args=ns.build_arg, trace_scopes=ns.trace_scope,
            trace_off_rules=ns.trace_off_rule, trace_depth=ns.trace_depth,
            pythonpath=[str(Path.cwd()), *ns.pythonpath],
        ),
    )
    return _finish(runner.Pipeline(cfg).execute())
