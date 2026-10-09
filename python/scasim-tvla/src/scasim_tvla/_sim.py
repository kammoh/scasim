"""Simulator worker: `python -m scasim_tvla._sim build|test SPEC.json`.

The runner starts one worker for the build and one for each batch. A worker has its own
process, so its `Runner`, its `os.environ`, and its CPU time belong to one job. The parent has
already removed the variables that would override the settings (see `runner.clean_env`).

`test` writes `sim-result.json` in the test directory. It records what cocotb reported in
`results.xml`. The exit code of `Runner.test()` is not enough: a failing test can exit with 0 or
raise `SystemExit(1)`, and a missing test filter exits with 0 (E12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _build(spec: dict) -> int:
    from cocotb_tools.runner import get_runner

    get_runner("verilator").build(
        sources=spec["sources"], hdl_toplevel=spec["toplevel"], build_dir=spec["build_dir"],
        build_args=spec["build_args"], waves=True, clean=True, always=True,
    )
    return 0


def _test(spec: dict) -> int:
    from cocotb_tools.runner import get_results, get_runner

    test_dir = Path(spec["test_dir"])
    results_xml = test_dir / "results.xml"
    result: dict = {"exit": 0, "tests": 0, "fails": 0, "error": None}
    profile = bool(spec.get("profile"))
    try:
        get_runner("verilator").test(
            hdl_toplevel=spec["toplevel"], hdl_toplevel_lang="verilog",
            test_module=spec["test_module"], testcase=spec.get("testcase"),
            build_dir=spec["build_dir"], test_dir=str(test_dir), seed=spec["seed"], waves=True,
            test_args=["--trace-file", spec["trace_file"]], extra_env=spec["env"],
            results_xml=str(results_xml),
        )
    except SystemExit as exc:
        result["exit"] = exc.code
    except Exception as exc:  # noqa: BLE001 - report every failure of the run
        result["error"] = f"{type(exc).__name__}: {exc}"
    if results_xml.exists():
        result["tests"], result["fails"] = get_results(results_xml)
    elif result["error"] is None:
        result["error"] = "the simulation wrote no results.xml"
    if profile:
        # Imported only here: a run without --profile loads nothing for it.
        import resource

        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["cpu"] = {"user": child.ru_utime, "system": child.ru_stime}
        pstat = test_dir / "cocotb.pstat"
        if pstat.exists():
            import pstats

            with open(test_dir / "profile.txt", "w") as fh:
                pstats.Stats(str(pstat), stream=fh).sort_stats("cumulative").print_stats(25)
    (test_dir / "sim-result.json").write_text(json.dumps(result))
    return 0


def main(argv: list[str]) -> int:
    mode, spec_path = argv
    spec = json.loads(Path(spec_path).read_text())
    sys.path[:0] = spec.get("pythonpath", [])
    if mode == "build":
        return _build(spec)
    if mode == "test":
        return _test(spec)
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
