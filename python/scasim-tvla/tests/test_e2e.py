"""End-to-end tests: Verilator, cocotb 2.1, cocotbext-stream, scasim_tvla, and the tvla binary.

They skip without Verilator, cocotb 2.1, or a tvla binary. The design is `e2e/leaky.sv`: a masked
pipeline with a planted leak. The testbench is `e2e/tb_leaky.py`. Sample numbers: the segment
starts at an edge (bin 0). The driver sets `valid` at the next edge, and the next edge accepts it
(E_0, bin 2). The leak register loads at E_2 (bin 4) and clears at E_3 (bin 5).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from npz_util import flagged, members, read_array, read_f64
from scasim_tvla import cli, runner

E2E = Path(__file__).parent / "e2e"
CLOCK_ARGS = ["--clock", "leaky.clk", "--include", "scope:leaky",
              "--exclude", "signal:leaky.in_data", "--plot=false", "-d", "2"]
STAGE = 2
LEAK_BINS = [2 + STAGE, 3 + STAGE]  # the capture edge and the clear edge
SEED = 5
TESTS = 120

pytestmark = pytest.mark.usefixtures("sim_available")


class Runs:
    """Runs `scasim-tvla run` in a directory per name. Builds of one design are shared."""

    def __init__(self, root, tvla):
        self.root, self.tvla = Path(root), tvla
        self._builds = {}

    def out(self, name):
        return self.root / name

    def run(self, name, *, batches=4, jobs=2, keep="none", tests=TESTS, seed=SEED,
            build_args=(), tvla_args=CLOCK_ARGS, env=None, extra=(), expect=0, clean=True):
        out = self.out(name)
        key = tuple(build_args) + tuple(a for a in extra if a.startswith("--trace"))
        if key in self._builds and not (out / "build").exists():
            shutil.copytree(self._builds[key], out / "build")
        argv = ["run", "--sources", str(E2E / "leaky.sv"), "--toplevel", "leaky",
                "--test-module", "tb_leaky", "--testcase", "leak_test",
                "--pythonpath", str(E2E), "--batches", str(batches),
                "--tests-per-batch", str(tests), "--out", str(out), "--seed", str(seed),
                "--jobs", str(jobs), "--keep", *keep.split(), "--tvla", str(self.tvla),
                *[f"--build-arg={a}" for a in build_args], *extra, "--", *tvla_args]
        with mock.patch.dict(os.environ, env or {}):
            code = cli.main(argv)
        assert code == expect, f"{name}: exit {code}, expected {expect}"
        if key not in self._builds and (out / "build").exists():
            self._builds[key] = self.out("_build_" + str(len(self._builds)))
            shutil.copytree(out / "build", self._builds[key])
        return out

    def tvla_run(self, *args):
        done = subprocess.run([str(self.tvla), *args], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return done


@pytest.fixture(scope="module")
def runs(tmp_path_factory, tvla_bin):
    return Runs(tmp_path_factory.mktemp("e2e"), tvla_bin)


# The shell values below would override the runner's settings if the runner did not remove them.
HOSTILE_ENV = {"COCOTB_RANDOM_SEED": "1", "COCOTB_TEST_FILTER": "no_such_test",
               "SCASIM_TVLA_SEED": "99", "SCASIM_TVLA_BATCH": "wrong", "WAVES": "0"}


@pytest.fixture(scope="module")
def main_run(runs):
    """The planted leak, four batches, two jobs, waveforms kept, in a hostile environment."""
    return runs.run("main", keep="waveform", env=HOSTILE_ENV)


def report_arrays(out):
    return members(out / "report" / "t_values.npz"), members(out / "report" / "chi2.npz")


def test_the_planted_leak_is_found_at_the_right_samples(main_run):
    shape, _ = read_f64(main_run / "report" / "t_values.npz", "t_values")
    assert shape == (2, 8)
    hits = flagged(main_run / "report" / "t_values.npz")
    assert hits[1] == LEAK_BINS
    assert set(hits[2]) <= set(LEAK_BINS)


def test_the_run_ignores_hostile_shell_variables(main_run):
    for batch in sorted(p for p in main_run.iterdir() if p.name.startswith("b0")):
        meta = json.loads((batch / "meta.json").read_text())
        assert meta["batch"]["id"] == batch.name
        assert meta["batch"]["seeds"]["base"] == runner.batch_seed(SEED, batch.name)
        assert len(meta["segments"]) == TESTS


def test_keep_waveform_keeps_the_waveforms_the_caches_and_the_list(main_run):
    for b in ("b0000", "b0001", "b0002", "b0003"):
        assert (main_run / b / "tvla.fst").stat().st_size > 0
        assert (main_run / b / "statistics.bin").stat().st_size > 0
    assert (main_run / "meta.list").read_text().splitlines() == [
        f"b000{i}/meta.json" for i in range(4)]
    manifest = json.loads((main_run / "manifest.json").read_text())
    assert {r["state"] for r in manifest["batches"].values()} == {"cached"}
    assert "sim_cpu" not in manifest["batches"]["b0000"]
    assert not list(main_run.glob("b*/cocotb.pstat"))  # no profiling when --profile is off


def test_the_merged_result_equals_a_direct_run_over_the_waveforms(runs, main_run):
    direct = runs.out("direct")
    runs.tvla_run("--meta-list", str(main_run / "meta.list"), *CLOCK_ARGS,
                  "--ttest-output-dir", str(direct))
    merged_t, merged_chi = report_arrays(main_run)
    assert members(direct / "t_values.npz") == merged_t
    assert members(direct / "chi2.npz") == merged_chi


def test_one_job_and_two_jobs_give_identical_results(runs, main_run):
    one = runs.run("jobs1", jobs=1, keep="none")
    assert report_arrays(one) == report_arrays(main_run)
    # keep none: the metadata and the cache stay, the waveform goes
    for b in ("b0000", "b0003"):
        assert (one / b / "meta.json").exists() and (one / b / "statistics.bin").exists()
        assert not (one / b / "tvla.fst").exists()
    assert (one / "meta.list").read_text() == ""


def test_the_collect_and_merge_flow_equals_the_run(runs, main_run):
    flow = runs.out("flow")
    for b in ("b0000", "b0001", "b0002", "b0003"):
        (flow / b).mkdir(parents=True)
        for name in ("meta.json", "tvla.fst"):
            shutil.copy(main_run / b / name, flow / b / name)
    assert cli.main(["collect", str(flow), "--", *CLOCK_ARGS]) == 0
    assert cli.main(["merge", str(flow), "--tvla", str(runs.tvla)]) == 0
    assert report_arrays(flow) == report_arrays(main_run)


def test_shuffled_labels_are_not_flagged(runs, main_run):
    shuf = runs.out("shuffled")
    for b in ("b0000", "b0001", "b0002", "b0003"):
        (shuf / b).mkdir(parents=True)
        for name in ("meta.json", "tvla.fst"):
            shutil.copy(main_run / b / name, shuf / b / name)
    assert cli.main(["collect", str(shuf), "--", *CLOCK_ARGS, "--shuffle-labels", "11"]) == 0
    assert cli.main(["merge", str(shuf), "--tvla", str(runs.tvla)]) == 0
    hits = flagged(shuf / "report" / "t_values.npz")
    assert hits == {1: [], 2: []}


def test_a_design_without_the_leak_is_not_flagged(runs, main_run):
    out = runs.run("noleak", build_args=["-GLEAK=0"])
    assert flagged(out / "report" / "t_values.npz") == {1: [], 2: []}
    assert report_arrays(out) != report_arrays(main_run)


def test_a_leak_at_the_final_edge_is_inside_the_segment(runs, main_run):
    # The body returns after E_3, the edge that follows the last operation edge E_2.
    # The capture at E_2 is the last bin of the segment. The clear at E_3 is outside.
    inside = runs.run("final_edge", env={"TB_CYCLES": "3"})
    shape, _ = read_f64(inside / "report" / "t_values.npz", "t_values")
    assert shape == (2, 5)
    assert flagged(inside / "report" / "t_values.npz")[1] == [LEAK_BINS[0]]
    # Returning at E_2 itself would put the capture at the segment end, which is outside.
    outside = runs.run("final_edge_missed", env={"TB_CYCLES": "2"})
    shape, _ = read_f64(outside / "report" / "t_values.npz", "t_values")
    assert shape == (2, 4)
    assert flagged(outside / "report" / "t_values.npz") == {1: [], 2: []}


def test_a_failing_batch_is_excluded_and_keeps_its_waveform(runs):
    out = runs.run("failing", batches=3, expect=1, env={"TB_FAIL_BATCH": "b0001"})
    manifest = json.loads((out / "manifest.json").read_text())["batches"]
    assert manifest["b0001"]["state"] == "failed"
    assert manifest["b0000"]["state"] == manifest["b0002"]["state"] == "cached"
    assert (out / "b0001" / "tvla.fst").stat().st_size > 0
    assert json.loads((out / "b0001" / "meta.json").read_text())["batch"]["status"] == "diagnostic"
    assert (out / "b0001" / "diagnostic.txt").read_text().strip()
    assert not (out / "b0001" / "statistics.bin").exists()
    assert not (out / "b0000" / "tvla.fst").exists()
    log = (out / "report" / "tvla.log").read_text()
    assert "b0001" not in log and "b0000/statistics.bin" in log and "b0002/statistics.bin" in log


def test_a_rerun_skips_cached_batches_and_retries_failed_ones(runs):
    out = runs.run("rerun", batches=3, expect=1, env={"TB_FAIL_BATCH": "b0001"})
    stats_time = {b: (out / b / "statistics.bin").stat().st_mtime_ns for b in ("b0000", "b0002")}
    simulated, analyzed = [], []
    real_sim, real_an = runner.Pipeline.simulate_batch, runner.Pipeline.analyze_batch

    def spy_sim(self, batch, tests):
        simulated.append(batch)
        return real_sim(self, batch, tests)

    def spy_an(self, batch):
        analyzed.append(batch)
        return real_an(self, batch)

    with mock.patch.object(runner.Pipeline, "simulate_batch", spy_sim), \
            mock.patch.object(runner.Pipeline, "analyze_batch", spy_an):
        runs.run("rerun", batches=3)  # the testbench passes now
    assert [b for b in simulated if b != "probe"] == ["b0001"]
    assert analyzed == ["b0001"]
    assert {b: (out / b / "statistics.bin").stat().st_mtime_ns for b in stats_time} == stats_time
    manifest = json.loads((out / "manifest.json").read_text())["batches"]
    assert {r["state"] for r in manifest.values()} == {"cached"}
    # The retry used the same seed, so the batch equals the one from a clean run.
    reference = runs.run("rerun_clean", batches=3)
    assert report_arrays(out) == report_arrays(reference)
    simulated.clear()
    with mock.patch.object(runner.Pipeline, "simulate_batch", spy_sim):
        runs.run("rerun", batches=3)
    assert simulated == []


def test_no_analyze_keeps_the_waveforms_and_merge_finishes_the_job(runs, main_run):
    out = runs.run("noanalyze", extra=["--no-analyze"])
    assert not (out / "report").exists()
    assert all((out / f"b000{i}" / "tvla.fst").exists() for i in range(4))
    assert not list(out.glob("b*/statistics.bin"))
    assert cli.main(["collect", str(out), "--", *CLOCK_ARGS]) == 0
    assert cli.main(["merge", str(out), "--tvla", str(runs.tvla)]) == 0
    assert report_arrays(out) == report_arrays(main_run)


def test_profile_reports_the_simulator_cpu(runs):
    out = runs.run("profile", batches=2, extra=["--profile"])
    manifest = json.loads((out / "manifest.json").read_text())["batches"]
    for rec in manifest.values():
        assert rec["sim_cpu"]["user"] + rec["sim_cpu"]["system"] > 0
    assert (out / "b0000" / "profile.txt").read_text().strip()
    report = json.loads((out / "report" / "run.json").read_text())
    assert set(report["sim_cpu"]) == {"b0000", "b0001"}


def test_a_trace_off_rule_removes_a_scope_from_the_waveform(runs):
    out = runs.run("scoped", batches=1, keep="waveform", extra=["--trace-off-rule", "leaky.u_noise"])
    assert (out / "trace.vlt").read_text() == '`verilator_config\ntracing_off -scope "leaky.u_noise"\n'
    listing = runs.tvla_run("--meta-json", str(out / "b0000" / "meta.json"), "--list-signals").stdout
    assert "leaky.share" in listing or "leaky.leak" in listing
    assert "u_noise" not in listing


def test_a_rule_that_matches_nothing_stops_the_run_before_the_batches(runs, capsys):
    args = [a if a != "scope:leaky" else "scope:no_such_scope" for a in CLOCK_ARGS]
    runs.run("badrule", batches=2, tvla_args=args, expect=2)
    assert "matches no signal" in capsys.readouterr().err
    assert not list(runs.out("badrule").glob("b0*"))
    assert (runs.out("badrule") / "probe" / "meta.json").exists()  # kept for diagnosis


def test_a_changed_include_file_rebuilds_the_design(tmp_path, tvla_bin):
    src, header = tmp_path / "inc_top.sv", tmp_path / "defs.svh"
    src.write_text('`include "defs.svh"\nmodule inc_top(input logic clk, output logic [`W-1:0] q);\n'
                   "  always_ff @(posedge clk) q <= q + 1;\nendmodule\n")
    header.write_text("`define W 8\n")
    (tmp_path / "files.f").write_text(f"inc_top.sv\n+incdir+{tmp_path}\n")
    builds = []
    real = runner.Pipeline._run_worker

    def spy(self, mode, spec_path, log):
        builds.append(mode)
        return real(self, mode, spec_path, log)

    def build():
        cfg = runner.RunConfig(out=tmp_path / "out", batches=1, tests_per_batch=1, analyze=False,
                               sim=runner.SimSpec(sources=[str(tmp_path / "files.f")], toplevel="inc_top"))
        runner.Pipeline(cfg)._build()

    with mock.patch.object(runner.Pipeline, "_run_worker", spy):
        (tmp_path / "out").mkdir()
        build()
        build()
        assert builds == ["build"]  # the second call reuses the build
        header.write_text("`define W 16\n")  # only the included file changes
        build()
        assert builds == ["build", "build"]
    record = json.loads((tmp_path / "out" / "build" / "scasim-tvla-build.json").read_text())
    assert str(header.resolve()) in record["inputs"] or str(header) in record["inputs"]


def welch_order_1(shape, values, labels):
    """The order-1 t-value of each sample, for the labels 0 and 1 (pure Python).

    The variances are the biased ones (divided by the count), as in the Rust t-test.
    """
    rows, cols = shape
    t = []
    for j in range(cols):
        groups = {0: [], 1: []}
        for i in range(rows):
            if labels[i] in groups:
                groups[labels[i]].append(values[i * cols + j])
        mean = {c: sum(v) / len(v) for c, v in groups.items()}
        var = {c: sum((x - mean[c]) ** 2 for x in v) / len(v) for c, v in groups.items()}
        denominator = (var[0] / len(groups[0]) + var[1] / len(groups[1])) ** 0.5
        t.append((mean[0] - mean[1]) / denominator if denominator else float("nan"))
    return t


def test_keep_traces_keeps_the_traces_and_deletes_the_waveform(runs):
    out = runs.run("traces", batches=1, keep="traces")
    batch = out / "b0000"
    traces = batch / runner.TRACES_NAME
    assert traces.stat().st_size > 0
    assert (batch / "statistics.bin").exists() and (batch / "meta.json").exists()
    assert not (batch / "tvla.fst").exists()
    manifest = json.loads((out / "manifest.json").read_text())["batches"]["b0000"]
    assert manifest["traces"] == f"b0000/{runner.TRACES_NAME}"
    assert manifest["traces_bytes"] == traces.stat().st_size
    # The file: 120 segments, 8 samples, the raw labels, and the meta entry.
    shape, values = read_array(traces, "t_0", "<u4")
    assert shape == (TESTS, 8)
    _, labels = read_array(traces, "labels", "<u2")
    _, ids = read_array(traces, "segment_ids", "<u8")
    assert sorted(set(labels)) == [0, 1] and len(ids) == TESTS
    _, raw = read_array(traces, "meta.json", "|u1")
    meta = json.loads(bytes(raw))
    assert meta["samples"] == 8 and meta["segments"] == TESTS and meta["batch_id"] == "b0000"
    assert [c["name"] for c in meta["channels"]] == ["total"]
    # The t-values of the traces equal the merged report of this one batch.
    _, reported = read_f64(out / "report" / "t_values.npz", "t_values")
    recomputed = welch_order_1(shape, values, labels)
    assert recomputed == pytest.approx(reported[:8], rel=1e-8, abs=1e-10)


def test_keep_traces_and_waveform_keep_both_and_the_file_equals_a_direct_tvla_run(runs):
    out = runs.run("both", batches=1, keep="traces waveform", extra=["--traces-channels", "total"])
    batch = out / "b0000"
    assert (batch / "tvla.fst").exists() and (batch / runner.TRACES_NAME).exists()
    direct = runs.out("both_direct.npz")
    runs.tvla_run("--meta-json", str(batch / "meta.json"), *CLOCK_ARGS, "--traces-out", str(direct),
                  "--ttest-output-dir", str(runs.out("both_direct_out")))
    kept = members(batch / runner.TRACES_NAME)
    again = members(direct)
    for name in ("t_0", "labels", "groups", "segment_ids"):
        assert kept[name] == again[name], name
