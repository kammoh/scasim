#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import sys
from cocotb.runner import (  # type: ignore
    VHDL,
    Simulator,
    Verilator,
    Verilog,
    get_runner,
)
from copy import deepcopy
from dataclasses import dataclass
from joblib import Parallel, delayed  # type: ignore
from logging import getLogger
from pathlib import Path
from typing import Mapping
from xml.etree import ElementTree

logger = getLogger(__name__)


@dataclass
class TestCase:
    name: str
    classname: str
    file: str
    lineno: str
    time: float
    sim_time_ns: float | None
    ratio_time: float
    status: str


@dataclass
class TestSuite:
    random_seed: int
    test_cases: list["TestCase"]
    errors: int
    failures: int
    total_sim_time_ns: float


def parse_results(results_xml_file) -> list[TestSuite]:
    tree = ElementTree.parse(results_xml_file)
    results = []
    for ts in tree.iter("testsuite"):
        random_seed = ts.get("random_seed")
        test_cases = []
        num_errors = 0
        num_failures = 0
        num_skipped = 0
        total_sim_time_ns = 0.0
        for tc in ts.iter("testcase"):
            sim_time_ns_str = tc.get("sim_time_ns")
            if sim_time_ns_str is None:
                sim_time_ps_str = tc.get("sim_time_ps")
                if sim_time_ps_str is not None:
                    try:
                        sim_time_ns = float(sim_time_ps_str) / 1e3
                    except ValueError:
                        sim_time_ns = 0
                else:
                    sim_time_ns = 0
            else:
                try:
                    sim_time_ns = float(sim_time_ns_str)
                except ValueError:
                    sim_time_ns = 0
            time_s = float(tc.get("time") or 0)
            failures = [e.tag.upper() for e in tc]
            test_cases.append(
                TestCase(
                    name=tc.get("name", "???"),
                    classname=tc.get("classname", "???"),
                    file=tc.get("file", "???"),
                    lineno=tc.get("lineno", "???"),
                    time=time_s,
                    sim_time_ns=sim_time_ns,
                    ratio_time=sim_time_ns / time_s if time_s > 0 else 0,
                    status=", ".join(failures) or "PASSED",
                )
            )
            num_errors += len(list(filter(lambda e: e == "ERROR", failures)))
            num_failures += len(list(filter(lambda e: e == "FAILURE", failures)))
            num_skipped += len(list(filter(lambda e: e == "SKIPPED", failures)))
            total_sim_time_ns += sim_time_ns
        results.append(
            TestSuite(
                random_seed=int(random_seed or "-1"),
                test_cases=test_cases,
                errors=num_errors,
                failures=num_failures,
                total_sim_time_ns=total_sim_time_ns,
            )
        )
    return results


def build_simulation(
    sources: list[str] | list[Path],
    top: str,
    build_dir: str | Path,
    sim: str = "verilator",
    build: bool = True,
    waves: bool | None = None,
    trace_ext="fst",
    verbose: bool = False,
):
    build_args: list[str | VHDL | Verilog] = []

    if waves is None:
        if os.getenv("WAVES") in ("1", "true", "True", "TRUE", "on", "On", "ON"):
            waves = True
        else:
            waves = False

    if sim == "verilator":
        cflags = "-march=native -mtune=native"
        env_cflags = os.environ.get("CPPFLAGS")
        if env_cflags:
            cflags += f" {env_cflags}"
        env_cflags = os.environ.get("CFLAGS")
        if env_cflags:
            cflags += f" {env_cflags}"
        # os.environ["CFLAGS"] = cflags
        # os.environ["CXXFLAGS"] = cflags
        verilate_flags = [
            "-Wno-fatal",
            "-Wno-lint",
            "-Wno-style",
            "-Wno-UNOPTFLAT",
            "-O3",
            "--x-assign",
            "fast",
            "--x-initial",
            "fast",
            "-j",
            "0",
            # "--flatten",
        ]
        # verilate_flags += ["--assert"]
        verilate_flags += ["-CFLAGS", f"{cflags}"]

        if waves:
            verilate_flags += [
                "--trace-underscore",
                "--trace-structs",
                "--trace-max-array",
                "16384",
                "--trace-max-width",
                "16384",
                "--trace-threads",
                "4",
            ]
            if trace_ext == "fst":
                verilate_flags += ["--trace-fst"]
            elif trace_ext == "vcd":
                verilate_flags += ["--trace-vcd"]
            elif trace_ext == "saif":
                verilate_flags += ["--trace-saif"]

        for flag in verilate_flags:
            build_args.append(Verilog(flag))

    runner = get_runner(sim)
    try:
        runner.build(
            sources=[str(src) for src in sources],
            hdl_toplevel=top,
            clean=build,
            always=build,
            waves=waves,
            verbose=verbose,
            build_args=build_args,
            build_dir=build_dir,
            # parameters={},
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building the design {top}: {e}")
        exit(1)
    return runner


def run_test(
    runner: Simulator,
    top: str,
    test_modules: list[str],
    test_cases: list[str],
    build_dir: str | Path,
    test_dir: str | Path,
    hdl_toplevel_lang: str = "verilog",
    waves: bool | None = None,
    trace_filename: str | None = None,
    seed: int | str | None = None,
    extra_env: Mapping[str, str] | None = None,
    verbose: bool = False,
):
    test_args: list[str] = []

    if waves:
        print("Enabling waveform generation")
        if isinstance(runner, Verilator) and trace_filename:
            test_args += ["--trace-file", trace_filename]

    print(
        f"Running tests for {top} test_modules={test_modules} trace={trace_filename if waves else False} seed={seed}"
    )

    results_xml_file = runner.test(
        test_module=test_modules,
        testcase=test_cases,
        hdl_toplevel=top,
        hdl_toplevel_lang=hdl_toplevel_lang,
        waves=waves,
        seed=seed,
        build_dir=build_dir,
        test_dir=test_dir,
        extra_env=extra_env or {},
        # parameters={},
        test_args=test_args,
        verbose=verbose,
    )
    print(f"Results file: {results_xml_file}")

    if results_xml_file and results_xml_file.exists():
        print(parse_results(results_xml_file))
    else:
        print("No results file found")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "filelist",
        type=Path,
        help="path to filelist.f",
    )
    argparser.add_argument(
        "--top",
        type=str,
        # required=True,
        help="top module name",
    )
    argparser.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose output",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for the tests",
    )
    argparser.add_argument(
        "--test-module",
        type=str,
        required=True,
        help="list of test modules to run (default: [top_tb])",
    )
    argparser.add_argument(
        "--test-cases",
        type=str,
        required=True,
        nargs="+",
        help="list of test cases to run (default: all test cases in the test modules)",
    )
    argparser.add_argument(
        "--build",
        action="store_true",
        help="whether to build the design before running tests (default: True)",
    )
    argparser.add_argument(
        "--test-root",
        type=Path,
        default=None,
    )
    argparser.add_argument(
        "--num-tests",
        type=int,
        default=1000,
        help="number of tests to run",
    )
    argparser.add_argument(
        "--parallel-jobs",
        "-j",
        type=int,
        default=0,
        help="number of parallel jobs to run (default: 0, which means use all available CPUs)",
    )

    argparser.add_argument(
        "--single-test",
        action="store_true",
    )
    argparser.add_argument(
        "--trace-filename",
        type=str,
        default="tvla.fst",
        help="name of the trace file to generate (default: tvla.fst)",
    )
    argparser.add_argument(
        "--multi",
        "--multiple-runs",
        type=int,
        help="run the test multiple times with different seeds",
    )
    argparser.add_argument(
        "--multi-dir-suffix",
        type=str,
        default="multi",
        help="suffix for the directory where multiple runs will be stored (default: multi)",
    )

    args = argparser.parse_args()

    # if there's a command line argument, assume it's a XEDA YAML design file and run run_test_xeda_design

    filelist = args.filelist
    if not isinstance(filelist, Path):
        filelist = Path(filelist)
    if not filelist.exists():
        print(f"Error: file list {filelist} does not exist")
        exit(1)
    sources_root = filelist.parent
    if filelist.suffix == ".f":
        with open(filelist, encoding="utf-8") as f:
            sources_lines = [line.strip() for line in f.readlines()]
            sources = [sources_root / src for src in sources_lines if src]
    elif filelist.suffix == ".json":
        import json

        with open(filelist, encoding="utf-8") as f:
            sources_data = json.load(f)
            if not isinstance(sources_data, dict):
                print(f"Error: expected a JSON object in {filelist}, got {type(sources_data)}")
                exit(1)
            if "rtl" in sources_data:
                sources_data = sources_data["rtl"]
            assert isinstance(
                sources_data, dict
            ), f"Expected a JSON object in {filelist}, got {type(sources_data)}"
            if "sources" not in sources_data:
                print(f"Error: expected a 'sources' key in the JSON file {filelist}")
                exit(1)
            sources = [sources_root / src for src in sources_data["sources"]]
            if not args.top:
                if "top" in sources_data:
                    args.top = sources_data["top"]
                else:
                    print(f"Error: no top module specified and no 'top' key found in {filelist}")
                    exit(1)

    assert sources, "No sources found in the file list"
    for src in sources:
        if not src.exists():
            print(f"Error: source file {src} does not exist")
            exit(1)

    test_root = args.test_root or Path.cwd() / "tvla_run" / args.top
    build_dir: Path = test_root / "sim_build"

    if not build_dir.exists():
        # build_dir.mkdir(parents=True, exist_ok=True)
        args.build = True
    else:
        sim_exe_path = build_dir / args.top
        if not sim_exe_path.exists():
            print(f"Simulation executable {sim_exe_path} does not exist, forcing build.")
            args.build = True
        else:
            sources_last_modified = max(src.stat().st_mtime for src in sources)
            build_files = list(build_dir.glob("**/*"))  # get all files in the build directory
            if not build_files:
                args.build = True
            else:
                build_dir_last_modified = max(f.stat().st_mtime for f in build_files)
                if sources_last_modified > build_dir_last_modified:
                    print("Sources have been modified since the last build. Forcing clean+build.")
                    args.build = True

    trace_filename = args.trace_filename

    if trace_filename:
        trace_ext = trace_filename.split(".")[-1]
    else:
        trace_ext = "vcd"

    if args.build:
        runner = build_simulation(
            sources=sources,
            top=args.top,
            build_dir=build_dir,
            sim="verilator",
            waves=True,
            trace_ext=trace_ext,
            verbose=args.verbose,
        )
    else:
        runner = get_runner("verilator")
        runner.hdl_toplevel_lang = "verilog"

    meta_filename = "meta.json.gz"

    extra_env = {"TVLA": "1"}
    extra_env["NUM_TESTS"] = str(args.num_tests)
    extra_env["TRACE_FILENAME"] = trace_filename
    extra_env["TVLA_META_FILENAME"] = meta_filename

    test_cases = args.test_cases
    if test_cases is None:
        logger.error("No test cases specified. Please provide test cases using --test-cases.")
        exit(1)
    print(f"Test cases: {test_cases}")
    if not isinstance(test_cases, list):
        test_cases = [test_cases]
    assert (
        len(test_cases) == 1
    ), f"TVLA tests should only run one test case at a time (test_cases={test_cases}"
    test_case = test_cases[0]

    assert args.test_module
    if args.test_module.endswith(".py") and args.test_module.startswith("./"):
        # translate path to module name
        args.test_module = args.test_module[2:-3].replace(os.sep, ".")

    # constant_rand = os.getenv("TVLA_DISABLE_RANDOM", "0").lower() in ("1", "true", "yes")

    sys.path.append(os.getcwd())

    def run_vs_no_random(seed, constant_rand, extra_env):
        run_extra_env = deepcopy(extra_env)
        if constant_rand:
            print("Disabling randomization for TVLA tests")
            run_extra_env["TVLA_DISABLE_RANDOM"] = "1"
        run_id = f"{seed:08x}"
        if constant_rand:
            run_id += "_no_random"
        test_dir = test_root / test_case / run_id

        print(f"Using seed: {seed}, run_id: {run_id}, test_dir: {test_dir}")

        run_test(
            runner,
            top=args.top,
            test_modules=args.test_module,
            test_cases=test_cases,
            build_dir=build_dir,
            test_dir=test_dir,
            waves=True,
            trace_filename=trace_filename,
            verbose=args.verbose,
            seed=seed,
            extra_env=run_extra_env,
        )

    def run_multi(id, multi_test_dir: Path):
        assert (
            args.seed is None
        ), "Multi tests should not use a fixed seed, it will be generated randomly for each run"

        meta_file = None
        while True:
            seed = random.randint(0, 2**31 - 1)
            run_id = f"{id}{seed:08x}"

            test_dir = multi_test_dir / run_id

            meta_file = test_dir / meta_filename
            if not meta_file.exists():
                break
        assert meta_file

        print(f"Running multi test {id} in {test_dir} with seed {seed}")

        run_test(
            runner,
            top=args.top,
            test_modules=args.test_module,
            test_cases=test_cases,
            build_dir=build_dir,
            test_dir=test_dir,
            waves=True,
            trace_filename=trace_filename,
            verbose=args.verbose,
            seed=seed,
            extra_env=extra_env,
        )
        if not meta_file.exists():
            print(f"Error: meta file {meta_file} does not exist after running test {id}")
            return None
        return meta_file

    if args.parallel_jobs < 1:
        args.parallel_jobs = os.cpu_count() or 8

    if args.multi:
        try:
            num_runs = int(args.multi)
            assert num_runs > 0, "Number of runs must be a positive integer"
        except ValueError:
            print(f"Error: NUM_RUNS must be a positive integer, got {args.multi[0]}")
            exit(1)
        dir_suffix = args.multi_dir_suffix
        assert dir_suffix, "Directory suffix for multi tests cannot be empty"
        multi_test_top_dir = test_root / test_case / dir_suffix
        if not multi_test_top_dir.exists():
            multi_test_top_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(
                f"Warning: multi test directory {multi_test_top_dir} already exists. Results will be appended to the existing ones."
            )
        n_jobs = min(num_runs, args.parallel_jobs)
        meta_files = Parallel(n_jobs=n_jobs)(
            delayed(run_multi)(i, multi_test_top_dir) for i in range(num_runs)
        )
        if any(f is None for f in meta_files):
            print("Warning: Some meta files were not created during the multi test runs")
        # convert paths to relative to multi_test_top_dir
        meta_files = [f.relative_to(multi_test_top_dir) for f in meta_files if f is not None]
        if not meta_files:
            print("No meta files were created during the multi test runs")
            exit(1)
        print(f"Meta files:\n{"\n".join(str(f) for f in meta_files)}\n")
        # append all meta files to a meta.list file in multi_test_top_dir
        meta_list_file = multi_test_top_dir / "meta.list"
        with open(meta_list_file, "a", encoding="utf-8") as f:
            for meta_file in meta_files:
                f.write(f"{meta_file}\n")
    else:
        if args.seed is None:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = int(args.seed) & 0x7FFF_FFFF  # ensure seed is a 31-bit integer

        constant_rand_confs = [False, True]
        if args.single_test:
            constant_rand_confs = [False]

        n_jobs = min(len(constant_rand_confs), args.parallel_jobs)
        Parallel(n_jobs=n_jobs)(
            delayed(run_vs_no_random)(seed, constant_rand, extra_env)
            for constant_rand in constant_rand_confs
        )
