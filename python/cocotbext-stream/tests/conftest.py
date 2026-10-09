"""Build the Verilator test design once and run the cocotb test module against it."""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

HERE = Path(__file__).parent

# The runner passes the parent sys.path to the simulator as PYTHONPATH.
TB_DIR = str(HERE / "tb")
if TB_DIR not in sys.path:
    sys.path.insert(0, TB_DIR)


def sim_available() -> tuple[bool, str]:
    if shutil.which("verilator") is None:
        return False, "verilator not found on PATH"
    try:
        from cocotb_tools.runner import get_runner  # noqa: F401
    except ImportError:
        return False, "cocotb 2.1 or later is not installed"
    return True, ""


class SimRun:
    """Result of one cocotb run: a map from test name to None (pass) or a failure text."""

    def __init__(self, results: dict[str, str | None], skipped: set[str]):
        self.results = results
        self.skipped = skipped


def run_cocotb(tmp_path: Path, test_module: str, toplevel: str = "stream_top") -> SimRun:
    from cocotb_tools.runner import get_runner

    runner = get_runner("verilator")
    sources = [HERE / "hdl" / name for name in ("stream_fifo.sv", "stream_pipe.sv", "stream_reorder.sv", "stream_top.sv")]
    runner.build(
        sources=sources,
        hdl_toplevel=toplevel,
        build_dir=tmp_path / "build",
        build_args=["-Wno-fatal"],
        timescale=("1ns", "1ps"),
        log_file=tmp_path / "build.log",
    )
    results_xml = tmp_path / "run" / "results.xml"
    try:
        runner.test(
            test_module=test_module,
            hdl_toplevel=toplevel,
            build_dir=tmp_path / "build",
            test_dir=tmp_path / "run",
            results_xml=results_xml,
            seed=1234,
            log_file=tmp_path / "sim.log",
        )
    except SystemExit as exc:  # the runner exits when a cocotb test fails
        if not results_xml.exists():
            pytest.fail(f"simulator exited with {exc.code} and wrote no results; see {tmp_path / 'sim.log'}")
    results: dict[str, str | None] = {}
    skipped: set[str] = set()
    for case in ET.parse(results_xml).getroot().iter("testcase"):
        name = case.get("name")
        failure = case.find("failure")
        error = case.find("error")
        if case.find("skipped") is not None:
            skipped.add(name)
        problem = failure if failure is not None else error
        results[name] = None if problem is None else (problem.get("message") or "") + (problem.text or "")
    return SimRun(results, skipped)


@pytest.fixture(scope="session")
def stream_sim(tmp_path_factory) -> SimRun:
    ok, why = sim_available()
    if not ok:
        pytest.skip(why)
    return run_cocotb(tmp_path_factory.mktemp("stream_sim"), "tb_stream")
