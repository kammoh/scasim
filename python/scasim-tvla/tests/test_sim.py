"""Simulator tests: cocotb 2.1 and Verilator. Skipped when one is missing."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

cocotb = pytest.importorskip("cocotb")
if tuple(int(p) for p in cocotb.__version__.split(".")[:2]) < (2, 1):
    pytest.skip("needs cocotb 2.1 or later", allow_module_level=True)
if shutil.which("verilator") is None:
    pytest.skip("needs Verilator", allow_module_level=True)

from scasim_tvla.meta import read_meta  # noqa: E402

HERE = Path(__file__).parent

FIXTURE = HERE.parents[2] / "tests" / "data" / "scasim_meta_v1"
PERIOD = 10000


def fst_changes(fst: Path, signal: str) -> set[int]:
    """Times (in FST ticks) at which `signal` changes value, found with `fst2vcd`."""
    if shutil.which("fst2vcd") is None:
        pytest.skip("needs fst2vcd")
    vcd = fst.with_suffix(".vcd")
    subprocess.run(["fst2vcd", "-f", str(fst), "-o", str(vcd)], check=True, capture_output=True)
    text = vcd.read_text()
    ids = set(re.findall(rf"\$var \w+ \d+ (\S+) {signal}(?: \[\d+:\d+\])? \$end", text))
    assert ids
    times, now = set(), None
    for line in text.split("$enddefinitions")[1].splitlines():
        if line.startswith("#"):
            now = int(line[1:])
        elif line[:1] == "b" and line.split()[-1] in ids:
            times.add(now)
    return times


def test_times_and_metadata(run_tb):
    out, fails = run_tb("basic")
    assert fails == 0
    obs = json.loads((out / "observed.json").read_text())
    meta = read_meta(out / "meta.json")
    assert obs["precision"] == -12
    assert meta["time"] == {"mantissa": 1, "exponent": -12}
    assert meta["batch"]["status"] == "committed"
    assert meta["batch"]["id"] == "b0001"
    assert meta["waveform"] == "tvla.fst"
    assert meta["design"]["toplevel"] == "tiny"
    listed = [s for s in obs["segments"] if not s["warmup"]]
    assert [s["id"] for s in meta["segments"]] == [1, 2, 3, 4, 5, 6]
    assert [(s["start"], s["end"], s["label"]) for s in meta["segments"]] == [
        (s["start"], s["end"], s["label"]) for s in listed
    ]
    for s in meta["segments"]:
        assert s["end"] - s["start"] == 2 * PERIOD or s["end"] - s["start"] == PERIOD
    assert all(a["end"] == b["start"] for a, b in zip(meta["segments"], meta["segments"][1:]))
    assert (out / "tvla.fst").exists()


def test_same_seed_same_batch(run_tb):
    a, _ = run_tb("basic", seed=5, batch="b7")
    b, _ = run_tb("basic", seed=5, batch="b7")
    c, _ = run_tb("basic", seed=5, batch="b8")
    d, _ = run_tb("basic", seed=6, batch="b7")

    def key(out):
        o = json.loads((out / "observed.json").read_text())["segments"]
        return [(s["label"], s["x"]) for s in o]

    assert key(a) == key(b)
    assert key(a) != key(c)
    assert key(a) != key(d)
    assert read_meta(a / "meta.json")["batch"]["seeds"] == read_meta(b / "meta.json")["batch"]["seeds"]


def test_test_count_from_environment(run_tb):
    out, _ = run_tb("basic", tests=3)
    assert len(read_meta(out / "meta.json")["segments"]) == 3


def test_diagnostic_on_failing_test_with_context(run_tb):
    out, fails = run_tb("fail_with")
    assert fails == 1
    meta = read_meta(out / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert [s["id"] for s in meta["segments"]] == [1, 2]  # segment 3 was open


def test_diagnostic_on_timeout_cancel(run_tb):
    out, fails = run_tb("timeout_with")
    assert fails == 1
    meta = read_meta(out / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert len(meta["segments"]) >= 1


def test_bare_loop_failure_gives_diagnostic(run_tb):
    """Without `with tvla:` cocotb 2.1 still closes the generator, so the finally block runs."""
    out, fails = run_tb("fail_bare")
    assert fails == 1
    meta = read_meta(out / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert [s["id"] for s in meta["segments"]] == [1, 2]


def test_boundary_with_activity_at_the_final_edge(run_tb):
    out, fails = run_tb("boundary")
    assert fails == 0
    obs = json.loads((out / "observed.json").read_text())["segments"]
    meta = {s["id"]: s for s in read_meta(out / "meta.json")["segments"]}
    changes = fst_changes(out / "tvla.fst", "q")
    modes = set()
    for s in obs:
        if s["warmup"]:
            continue
        m = meta[s["id"]]
        assert s["edge"] in changes  # q really changes at the operation edge
        assert m["start"] == s["start"]
        modes.add(s["mode"])
        if s["mode"] == "A":
            assert m["end"] == s["edge"]  # half-open: the edge is outside this segment
        elif s["mode"] == "B":
            assert m["end"] == s["edge"] + PERIOD  # extend(1): the edge is inside
            assert s["start"] <= s["edge"] < m["end"]
        else:
            assert m["end"] == s["edge"] + 1  # end_at(edge + 1): the edge is inside
            assert s["start"] <= s["edge"] < m["end"]
    assert modes == {"A", "B", "C"}


@pytest.mark.skipif(not os.environ.get("SCASIM_TVLA_WRITE_FIXTURE"), reason="opt-in")
def test_make_fixture(run_tb):
    """Run with SCASIM_TVLA_WRITE_FIXTURE=1 to write tests/data/scasim_meta_v1/."""
    out, fails = run_tb("basic", seed=2026, batch="b0001")
    assert fails == 0
    assert (out / "tvla.fst").stat().st_size < 50_000
    FIXTURE.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "meta.json", FIXTURE / "meta.json")
    shutil.copy(out / "tvla.fst", FIXTURE / "tvla.fst")

