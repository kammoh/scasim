"""The batch pipeline with a fake simulator and a fake tvla. No Verilator needed."""

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest
from unittest.mock import patch as _patch

from scasim_tvla import _args, cli, runner
from scasim_tvla.meta import MetaWriter

FAKE = Path(__file__).with_name("fake_tvla.py")
TVLA_ARGS = ["--clock", "top.clk", "--include", "scope:top", "-d", "2", "--plot=false"]


class FakeBin:
    def __init__(self, path, log):
        self.path, self.log = path, log

    def __str__(self):
        return str(self.path)


@pytest.fixture
def fake_tvla(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "tvla"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "tvla.log"
    monkeypatch.setenv("FAKE_TVLA_LOG", str(log))
    return FakeBin(exe, log)


def calls(fake, flag):
    if not fake.log.exists():
        return []
    rows = [json.loads(line) for line in fake.log.read_text().splitlines()]
    return [r for r in rows if flag in r]


class FakeSim:
    """Writes the files that a real batch writes. Fails the batches in `fail`."""

    def __init__(self, fail=(), out_root=None):
        self.fail = set(fail)
        self.calls = []
        self.envs = {}
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.max_waveforms = 0

    def __call__(self, pipe, batch, info):
        with self.lock:
            self.calls.append(batch)
            self.envs[batch] = dict(info["env"])
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(0.05)
        d = Path(info["dir"])
        w = MetaWriter(d / "meta.json", labels={0: "fixed", 1: "random"}, batch_id=batch,
                       waveform="tvla.fst")
        w.segment(0, 10, 0)
        w.segment(10, 20, 1)
        (d / "tvla.fst").write_bytes(b"x" * (1000 + len(batch)))
        n_waves = len(list(d.parent.glob("b*/tvla.fst")))
        with self.lock:
            self.running -= 1
            self.max_waveforms = max(self.max_waveforms, n_waves)
        if batch in self.fail:
            w.write_diagnostic()
            return runner.SimResult(tests=1, fails=1)
        w.commit()
        return runner.SimResult(tests=1, fails=0)


def config(tmp_path, fake, **kw):
    base = dict(out=tmp_path / "out", batches=4, tests_per_batch=2, seed=7, jobs=2,
                tvla=str(fake), tvla_args=list(TVLA_ARGS))
    base.update(kw)
    return runner.RunConfig(**base)


def run(tmp_path, fake, sim=None, **kw):
    sim = sim or FakeSim()
    report = runner.Pipeline(config(tmp_path, fake, **kw), simulate=sim).execute()
    return report, sim


def test_keep_none_keeps_the_cache_and_merges_in_id_order(tmp_path, fake_tvla):
    report, sim = run(tmp_path, fake_tvla, curve="every:2")
    out = tmp_path / "out"
    assert report.ok == ["b0000", "b0001", "b0002", "b0003"] and not report.failed
    assert report.exit_code == 0
    for b in report.ok:
        assert (out / b / "statistics.bin").read_text() == f"stats {b}\n"
        assert (out / b / "meta.json").exists()
        assert not (out / b / "tvla.fst").exists()
    merge = calls(fake_tvla, "--merge-stats")
    assert len(merge) == 1
    args = merge[0]
    files = args[args.index("--merge-stats") + 1 : args.index("--merge-stats") + 5]
    assert files == [str(out / b / "statistics.bin") for b in report.ok]
    assert args[args.index("--curve") + 1] == "every:2"
    assert args[args.index("--ttest-output-dir") + 1] == str(out / "report")
    assert "-d" in args and "--plot=false" in args
    for forbidden in ("--clock", "--include", "--meta-json", "--stats-out"):
        assert forbidden not in args
    assert (out / "report" / "merged.txt").read_text() == "".join(f"stats {b}\n" for b in report.ok)
    manifest = json.loads((out / "manifest.json").read_text())
    assert {r["state"] for r in manifest["batches"].values()} == {"cached"}
    assert manifest["batches"]["b0001"]["cache_bytes"] > 0
    assert (out / "meta.list").read_text() == ""
    assert json.loads((out / "report" / "run.json").read_text())["jobs"] == 2


def test_the_batch_call_gets_preprocessing_options_and_no_plots(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla)
    stats = calls(fake_tvla, "--stats-out")
    assert len(stats) == 4
    a = stats[0]
    assert a[a.index("--clock") + 1] == "top.clk"
    assert ["--include", "scope:top"] == a[a.index("--include") : a.index("--include") + 2]
    assert "--pool-groups" in a and "--plot=false" in a and "--chi2=false" in a
    assert "--curve" not in a


def test_keep_waveform_keeps_waveforms_and_lists_them(tmp_path, fake_tvla):
    report, _ = run(tmp_path, fake_tvla, keep="waveform")
    out = tmp_path / "out"
    assert all((out / b / "tvla.fst").exists() for b in report.ok)
    assert (out / "meta.list").read_text().splitlines() == [f"{b}/meta.json" for b in report.ok]


def test_no_analyze_only_simulates_and_keeps_waveforms(tmp_path, fake_tvla):
    report, _ = run(tmp_path, fake_tvla, analyze=False)
    out = tmp_path / "out"
    assert report.ok and report.report_dir is None
    assert all((out / b / "tvla.fst").exists() for b in report.ok)
    assert not (out / "report").exists()
    assert not calls(fake_tvla, "--stats-out") and not calls(fake_tvla, "--merge-stats")
    states = {r["state"] for r in json.loads((out / "manifest.json").read_text())["batches"].values()}
    assert states == {"simulated"}


def test_a_failing_batch_is_excluded_and_keeps_its_waveform(tmp_path, fake_tvla):
    report, _ = run(tmp_path, fake_tvla, sim=FakeSim(fail={"b0001"}))
    out = tmp_path / "out"
    assert set(report.failed) == {"b0001"} and report.exit_code == 1
    assert (out / "b0001" / "tvla.fst").exists()
    assert (out / "b0001" / "diagnostic.txt").read_text().strip()
    assert json.loads((out / "b0001" / "meta.json").read_text())["batch"]["status"] == "diagnostic"
    merge = calls(fake_tvla, "--merge-stats")[0]
    assert str(out / "b0001" / "statistics.bin") not in merge
    assert len([a for a in merge if a.endswith("statistics.bin")]) == 3
    assert not (out / "b0001" / "statistics.bin").exists()


def test_a_tvla_failure_fails_the_batch_and_keeps_the_waveform(tmp_path, fake_tvla, monkeypatch):
    monkeypatch.setenv("FAKE_TVLA_FAIL", "b0002")
    report, _ = run(tmp_path, fake_tvla)
    out = tmp_path / "out"
    assert set(report.failed) == {"b0002"}
    assert "boom" in report.failed["b0002"]
    assert (out / "b0002" / "tvla.fst").exists()


def test_the_test_exit_code_alone_does_not_decide(tmp_path, fake_tvla):
    class Silent(FakeSim):
        def __call__(self, pipe, batch, info):
            res = super().__call__(pipe, batch, info)
            if batch == "b0000":
                return runner.SimResult(exit_code=0, tests=0, fails=0)  # no test matched
            return res

    report, _ = run(tmp_path, fake_tvla, sim=Silent())
    assert "b0000" in report.failed


def test_a_rerun_skips_cached_batches_and_retries_failed_ones_with_the_same_seed(
        tmp_path, fake_tvla):
    sim1 = FakeSim(fail={"b0001"})
    run(tmp_path, fake_tvla, sim=sim1)
    seed1 = sim1.envs["b0001"]["SCASIM_TVLA_SEED"]
    stats_before = len(calls(fake_tvla, "--stats-out"))
    sim2 = FakeSim()
    report, _ = run(tmp_path, fake_tvla, sim=sim2)
    assert [b for b in sim2.calls if b != "probe"] == ["b0001"]
    assert sim2.envs["b0001"]["SCASIM_TVLA_SEED"] == seed1
    assert len(calls(fake_tvla, "--stats-out")) == stats_before + 1
    assert not report.failed and len(report.ok) == 4
    sim3 = FakeSim()
    run(tmp_path, fake_tvla, sim=sim3)
    assert sim3.calls == []  # not even the probe: nothing needs a simulation
    assert len(calls(fake_tvla, "--stats-out")) == stats_before + 1


def test_a_rerun_without_a_seed_uses_the_recorded_seed(tmp_path, fake_tvla):
    sim1 = FakeSim(fail={"b0000"})
    run(tmp_path, fake_tvla, sim=sim1, seed=None)
    sim2 = FakeSim()
    run(tmp_path, fake_tvla, sim=sim2, seed=None)
    assert sim1.envs["b0000"]["SCASIM_TVLA_SEED"] == sim2.envs["b0000"]["SCASIM_TVLA_SEED"]


def test_a_simulated_batch_with_its_waveform_is_only_analyzed_on_rerun(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla, analyze=False)
    sim = FakeSim()
    report, _ = run(tmp_path, fake_tvla, sim=sim)
    assert sim.calls == []
    assert len(calls(fake_tvla, "--list-signals")) == 2  # the selection is checked on a waveform
    assert len(report.ok) == 4 and len(calls(fake_tvla, "--stats-out")) == 4
    assert not any((tmp_path / "out" / b / "tvla.fst").exists() for b in report.ok)


def test_a_changed_configuration_runs_all_batches_again(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla)
    sim = FakeSim()
    run(tmp_path, fake_tvla, sim=sim, tests_per_batch=3)
    assert sorted(b for b in sim.calls if b != "probe") == ["b0000", "b0001", "b0002", "b0003"]


def test_a_changed_seed_runs_all_batches_again(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla, seed=7)
    sim = FakeSim()
    run(tmp_path, fake_tvla, sim=sim, seed=8)
    assert sorted(b for b in sim.calls if b != "probe") == ["b0000", "b0001", "b0002", "b0003"]
    assert sim.envs["b0000"]["SCASIM_TVLA_SEED"] == str(runner.batch_seed(8, "b0000"))
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["seed"] == 8


def test_a_rerun_with_fewer_batches_merges_only_those(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla, batches=4)
    report, _ = run(tmp_path, fake_tvla, batches=2)
    merge = calls(fake_tvla, "--merge-stats")[-1]
    assert [a for a in merge if a.endswith("statistics.bin")] == [
        str(tmp_path / "out" / b / "statistics.bin") for b in ("b0000", "b0001")]


def test_a_changed_selection_runs_all_batches_again(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla)
    sim = FakeSim()
    run(tmp_path, fake_tvla, sim=sim, tvla_args=["--clock", "top.clk", "--include", "scope:other"])
    assert len([b for b in sim.calls if b != "probe"]) == 4


def test_the_batch_environment(tmp_path, fake_tvla):
    _, sim = run(tmp_path, fake_tvla, design_random="on")
    env = sim.envs["b0002"]
    assert env["SCASIM_TVLA_BATCH"] == "b0002"
    assert env["SCASIM_TVLA_SEED"] == str(runner.batch_seed(7, "b0002"))
    assert env["SCASIM_TVLA_TESTS"] == "2"
    assert env["SCASIM_TVLA_WAVEFORM"] == "tvla.fst"
    assert env["SCASIM_TVLA_OUT"] == str((tmp_path / "out" / f"b0002.tmp-{os.getpid()}").resolve())
    assert env["SCASIM_TVLA_DESIGN_RANDOM"] == "on"
    assert "COCOTB_ENABLE_PROFILING" not in env


def test_profile_off_sets_and_imports_nothing(tmp_path, fake_tvla):
    watched = ("cocotb", "pstats", "cProfile")
    before = {m for m in watched if m in sys.modules}
    _, sim = run(tmp_path, fake_tvla)
    assert not any("COCOTB_ENABLE_PROFILING" in e for e in sim.envs.values())
    assert {m for m in watched if m in sys.modules} == before
    assert "sim_cpu" not in json.loads((tmp_path / "out" / "report" / "run.json").read_text())


def test_profile_on_sets_the_variable_and_reports_the_cpu(tmp_path, fake_tvla):
    class Cpu(FakeSim):
        def __call__(self, pipe, batch, info):
            res = super().__call__(pipe, batch, info)
            res.cpu = {"user": 1.5, "system": 0.25}
            return res

    _, sim = run(tmp_path, fake_tvla, sim=Cpu(), profile=True)
    assert all(e["COCOTB_ENABLE_PROFILING"] == "1" for e in sim.envs.values())
    data = json.loads((tmp_path / "out" / "report" / "run.json").read_text())
    assert data["sim_cpu"]["b0003"] == {"user": 1.5, "system": 0.25}


@pytest.mark.parametrize("jobs", [1, 2])
def test_the_disk_and_job_limits_hold(tmp_path, fake_tvla, monkeypatch, jobs):
    monkeypatch.setenv("FAKE_TVLA_SLEEP", "0.15")
    _, sim = run(tmp_path, fake_tvla, sim=FakeSim(), batches=8, jobs=jobs)
    assert sim.max_running <= jobs
    assert sim.max_waveforms <= 2 * jobs


def test_jobs_one_and_two_give_the_same_merge_input(tmp_path, fake_tvla):
    outs = []
    for jobs in (1, 2):
        sub = tmp_path / f"j{jobs}"
        sub.mkdir()
        run(sub, fake_tvla, jobs=jobs)
        outs.append((sub / "out" / "report" / "merged.txt").read_text())
    assert outs[0] == outs[1]


def test_the_probe_stops_the_run_on_a_selection_warning(tmp_path, fake_tvla, monkeypatch):
    monkeypatch.setenv("FAKE_TVLA_WARN", "1")
    sim = FakeSim()
    with pytest.raises(runner.RunnerError, match="matches no signal"):
        run(tmp_path, fake_tvla, sim=sim)
    assert sim.calls == ["probe"]
    assert (tmp_path / "out" / "probe").exists()  # kept for diagnosis


def test_the_probe_checks_the_clock_name(tmp_path, fake_tvla):
    with pytest.raises(runner.RunnerError, match="clock"):
        run(tmp_path, fake_tvla, tvla_args=["--clock", "top.nope"])


def test_the_probe_directory_is_removed_after_a_good_check(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla)
    assert not (tmp_path / "out" / "probe").exists()
    assert calls(fake_tvla, "--list-signals")[0].count("--include") == 1


def test_a_version_1_run_needs_a_clock(tmp_path, fake_tvla):
    with pytest.raises(_args.ArgsError, match="--clock"):
        runner.Pipeline(config(tmp_path, fake_tvla, tvla_args=["--include", "scope:top"]),
                        simulate=FakeSim())


def test_a_missing_tvla_binary_is_an_error_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("SCASIM_TVLA_BIN", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    sim = FakeSim()
    cfg = config(tmp_path, tmp_path / "none")
    cfg.tvla = None
    with pytest.raises(runner.RunnerError, match="tvla"):
        runner.Pipeline(cfg, simulate=sim).execute()
    assert sim.calls == []


# -- collect and merge --------------------------------------------------------------------


def make_batch(root, name, status="committed", wave=True, cache=False):
    d = root / name
    d.mkdir(parents=True)
    w = MetaWriter(d / "meta.json", labels={0: "a", 1: "b"}, batch_id=name,
                   waveform="tvla.fst" if wave else None)
    w.segment(0, 10, 0)
    w.commit() if status == "committed" else w.write_diagnostic()
    if wave:
        (d / "tvla.fst").write_bytes(b"x")
    if cache:
        (d / "statistics.bin").write_text(f"stats {name}\n")


def test_collect_orders_batch_directories_by_number(tmp_path):
    for name in ("b10", "b2", "b1"):
        make_batch(tmp_path, name)
    runner.collect(tmp_path, tvla_args=TVLA_ARGS)
    assert (tmp_path / "meta.list").read_text().splitlines() == [
        "b1/meta.json", "b2/meta.json", "b10/meta.json"]


def test_collect_writes_the_manifest_and_meta_list(tmp_path):
    make_batch(tmp_path, "b0", cache=True, wave=False)
    make_batch(tmp_path, "b1")
    make_batch(tmp_path, "b2", status="diagnostic")
    (tmp_path / "build").mkdir()
    (tmp_path / "notes").mkdir()
    m = runner.collect(tmp_path, tvla_args=TVLA_ARGS)
    states = {b: r["state"] for b, r in m.batches().items()}
    assert states == {"b0": "cached", "b1": "simulated", "b2": "failed"}
    assert (tmp_path / "meta.list").read_text().splitlines() == ["b1/meta.json"]
    assert json.loads((tmp_path / "manifest.json").read_text())["tvla_args"] == TVLA_ARGS


def test_merge_analyzes_waiting_batches_then_merges_in_order(tmp_path, fake_tvla):
    make_batch(tmp_path, "b0", cache=True, wave=False)
    make_batch(tmp_path, "b1")
    make_batch(tmp_path, "b2", status="diagnostic")
    runner.collect(tmp_path, tvla_args=TVLA_ARGS)
    report = runner.merge_dir(tmp_path, tvla=str(fake_tvla))
    assert report.failed.keys() == {"b2"}
    assert (tmp_path / "b1" / "statistics.bin").exists()
    assert (tmp_path / "b1" / "tvla.fst").exists()  # collect defaults to keep=waveform
    assert (tmp_path / "report" / "merged.txt").read_text() == "stats b0\nstats b1\n"


def test_merge_without_a_manifest_says_what_to_do(tmp_path, fake_tvla):
    with pytest.raises(runner.RunnerError, match="collect"):
        runner.merge_dir(tmp_path, tvla=str(fake_tvla))


# -- the command line --------------------------------------------------------------------


class _Captured(Exception):
    pass


def parse_run(monkeypatch, tmp_path, *extra):
    """The RunConfig that `scasim-tvla run` builds from the arguments."""
    seen = {}

    class Fake:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def execute(self):
            return runner.RunReport(ok=[], failed={}, report_dir=None)

    monkeypatch.setattr(runner, "Pipeline", Fake)
    code = cli.main(["run", "--sources", "x.sv", "--toplevel", "t", "--test-module", "m",
                     "--batches", "1", "--tests-per-batch", "1", "--out", str(tmp_path),
                     *extra])
    return code, seen.get("cfg")


def test_keep_takes_one_or_more_values(monkeypatch, tmp_path):
    _, cfg = parse_run(monkeypatch, tmp_path, "--keep", "traces")
    assert cfg.keep == "traces" and cfg.traces_channels == []
    _, cfg = parse_run(monkeypatch, tmp_path, "--keep", "waveform", "traces",
                       "--traces-channels", "total", "regex:a.*")
    assert cfg.keep == "traces+waveform" and cfg.traces_channels == ["total", "regex:a.*"]
    _, cfg = parse_run(monkeypatch, tmp_path)
    assert cfg.keep == "none"
    _, cfg = parse_run(monkeypatch, tmp_path, "--keep", "waveform")
    assert cfg.keep == "waveform"


@pytest.mark.parametrize("extra, reason", [
    (["--keep", "none", "traces"], "none"),
    (["--traces-channels", "total"], "--keep traces"),
    (["--keep", "waveform", "--traces-channels", "total"], "--keep traces"),
    (["--keep", "traces", "--no-analyze"], "analy"),
])
def test_keep_traces_options_that_do_not_fit_are_refused(monkeypatch, tmp_path, capsys, extra, reason):
    with pytest.raises(SystemExit) as exc:
        parse_run(monkeypatch, tmp_path, *extra)
    assert exc.value.code == 2
    assert reason in capsys.readouterr().err


def test_the_traces_options_belong_to_the_runner_not_to_tvla():
    for name in ("--traces-out", "--traces-channels"):
        with pytest.raises(_args.ArgsError):
            _args.partition(["--clock", "top.clk", name, "x"])


def traces_of(tmp_path, batch):
    return tmp_path / "out" / batch / runner.TRACES_NAME


def test_keep_traces_writes_the_file_next_to_the_cache_and_drops_the_waveform(tmp_path, fake_tvla):
    report, _ = run(tmp_path, fake_tvla, keep="traces", traces_channels=["total", "regex:a.*"])
    out = tmp_path / "out"
    assert not report.failed
    for b in report.ok:
        assert traces_of(tmp_path, b).read_text() == f"traces {b}\n"
        assert (out / b / "statistics.bin").exists()
        assert not (out / b / "tvla.fst").exists()
    stats = calls(fake_tvla, "--stats-out")
    for a in stats:
        assert a[a.index("--traces-out") + 1].endswith(runner.TRACES_NAME)
        i = a.index("--traces-channels")
        assert a[i + 1 : i + 3] == ["total", "regex:a.*"]
        assert a.index("--traces-channels") != a.index("--traces-out")
    assert "--traces-out" not in calls(fake_tvla, "--merge-stats")[0]
    assert "--traces-channels" not in calls(fake_tvla, "--merge-stats")[0]
    manifest = json.loads((out / "manifest.json").read_text())
    rec = manifest["batches"]["b0001"]
    assert rec["traces"] == f"b0001/{runner.TRACES_NAME}" and rec["traces_bytes"] > 0
    run_json = json.loads((out / "report" / "run.json").read_text())
    assert run_json["keep"] == "traces"
    assert run_json["batches"]["b0001"]["traces_bytes"] == rec["traces_bytes"]


def test_keep_traces_and_waveform_keep_both(tmp_path, fake_tvla):
    report, _ = run(tmp_path, fake_tvla, keep="traces+waveform")
    out = tmp_path / "out"
    assert all(traces_of(tmp_path, b).exists() and (out / b / "tvla.fst").exists()
               for b in report.ok)
    assert (out / "meta.list").read_text().splitlines() == [f"{b}/meta.json" for b in report.ok]
    a = calls(fake_tvla, "--stats-out")[0]
    assert "--traces-channels" not in a  # none were asked for


def test_without_keep_traces_tvla_gets_no_traces_options(tmp_path, fake_tvla):
    report, _ = run(tmp_path, fake_tvla, keep="waveform")
    assert not any(traces_of(tmp_path, b).exists() for b in report.ok)
    for a in calls(fake_tvla, "--stats-out"):
        assert "--traces-out" not in a and "--traces-channels" not in a


def test_a_failed_analysis_leaves_no_traces_file(tmp_path, fake_tvla, monkeypatch):
    monkeypatch.setenv("FAKE_TVLA_FAIL", "b0001")
    report, _ = run(tmp_path, fake_tvla, keep="traces")
    assert set(report.failed) == {"b0001"}
    assert not traces_of(tmp_path, "b0001").exists()
    assert traces_of(tmp_path, "b0000").exists()
    assert (tmp_path / "out" / "b0001" / "tvla.fst").exists()  # a failed batch keeps its waveform


def test_a_rerun_keeps_the_traces_and_simulates_a_batch_that_lost_its_file(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla, keep="traces")
    fake_tvla.log.unlink()
    _, sim = run(tmp_path, fake_tvla, keep="traces")
    assert sim.calls == [] and not calls(fake_tvla, "--stats-out")
    traces_of(tmp_path, "b0002").unlink()
    _, sim = run(tmp_path, fake_tvla, keep="traces")
    assert [b for b in sim.calls if b != "probe"] == ["b0002"]
    assert traces_of(tmp_path, "b0002").exists()


def test_switching_keep_traces_on_or_changing_the_channels_runs_all_batches_again(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla, keep="none")
    everything = ["b0000", "b0001", "b0002", "b0003"]
    _, sim = run(tmp_path, fake_tvla, keep="traces")
    assert sorted(b for b in sim.calls if b != "probe") == everything
    _, sim = run(tmp_path, fake_tvla, keep="traces", traces_channels=["total"])
    assert sorted(b for b in sim.calls if b != "probe") == everything
    _, sim = run(tmp_path, fake_tvla, keep="traces", traces_channels=["total"])
    assert sim.calls == []


def test_collect_and_merge_commands(tmp_path, fake_tvla, capsys):
    make_batch(tmp_path, "b0")
    assert cli.main(["collect", str(tmp_path), "--", *TVLA_ARGS]) == 0
    assert cli.main(["merge", str(tmp_path), "--tvla", str(fake_tvla)]) == 0
    assert "report in" in capsys.readouterr().out


def test_arguments_after_the_separator_go_to_tvla_and_bad_ones_are_usage_errors(
        tmp_path, capsys):
    make_batch(tmp_path, "b0")
    assert cli.main(["collect", str(tmp_path), "--", "--curve", "final"]) == 0  # stored only
    assert cli.main(["merge", str(tmp_path), "--tvla", "/nonexistent/tvla"]) == 2


# -- crash safety -----------------------------------------------------------------------


def test_a_batch_is_simulated_in_a_temporary_directory_and_then_renamed(tmp_path, fake_tvla):
    class Spy(FakeSim):
        def __call__(self, pipe, batch, info):
            self.dirs = getattr(self, "dirs", {})
            self.dirs[batch] = Path(info["dir"]).name
            return super().__call__(pipe, batch, info)

    sim = Spy()
    run(tmp_path, fake_tvla, sim=sim, keep="waveform")
    out = tmp_path / "out"
    assert ".tmp-" in sim.dirs["b0002"] and sim.dirs["b0002"].startswith("b0002")
    assert (out / "b0002" / "tvla.fst").exists()
    assert not list(out.glob("*.tmp-*"))


def test_a_crash_after_the_simulation_does_not_lose_the_batch(tmp_path, fake_tvla):
    real_set = runner.Manifest.set
    crashed = []

    def crashing_set(self, batch, **fields):
        if batch == "b0001" and fields.get("state") == "simulated" and not crashed:
            crashed.append(batch)
            raise RuntimeError("the run was killed here")
        return real_set(self, batch, **fields)

    sim1 = FakeSim()
    with _patch.object(runner.Manifest, "set", crashing_set):
        with pytest.raises(RuntimeError, match="killed"):
            run(tmp_path, fake_tvla, sim=sim1, jobs=1)
    out = tmp_path / "out"
    assert (out / "b0001" / "tvla.fst").exists()  # promoted: the data is there
    assert "state" not in json.loads((out / "manifest.json").read_text())["batches"]["b0001"]
    sim2 = FakeSim()
    report, _ = run(tmp_path, fake_tvla, sim=sim2, jobs=1)
    assert "b0001" not in sim2.calls  # recovered, not simulated again
    assert not report.failed and len(report.ok) == 4
    clean = tmp_path / "clean"
    clean.mkdir()
    run(clean, fake_tvla, jobs=1)
    assert (out / "report" / "merged.txt").read_text() == \
        (clean / "out" / "report" / "merged.txt").read_text()


def test_a_leftover_temporary_directory_is_removed_and_never_trusted(tmp_path, fake_tvla):
    out = tmp_path / "out"
    run(tmp_path, fake_tvla, batches=2)
    leftover = out / "b0001.tmp-4242"
    leftover.mkdir()
    (leftover / "tvla.fst").write_bytes(b"x")
    (leftover / "meta.json").write_text("{}")
    sim = FakeSim()
    run(tmp_path, fake_tvla, sim=sim, batches=2)
    assert not leftover.exists()
    assert sim.calls == []


def test_a_finished_directory_of_an_older_configuration_is_not_recovered(tmp_path, fake_tvla):
    run(tmp_path, fake_tvla, batches=2, analyze=False)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    for rec in manifest["batches"].values():
        rec.pop("state")
    (tmp_path / "out" / "manifest.json").write_text(json.dumps(manifest))
    sim = FakeSim()
    run(tmp_path, fake_tvla, sim=sim, batches=2, tests_per_batch=5)  # another configuration
    assert sorted(b for b in sim.calls if b != "probe") == ["b0000", "b0001"]


def test_collect_ignores_temporary_directories(tmp_path):
    make_batch(tmp_path, "b0")
    make_batch(tmp_path, "b1.tmp-7")
    runner.collect(tmp_path, tvla_args=TVLA_ARGS)
    assert (tmp_path / "meta.list").read_text().splitlines() == ["b0/meta.json"]


# -- build reuse ------------------------------------------------------------------------


class FakeBuild:
    """Replaces the build worker: writes the executable and a Verilator dependency file."""

    def __init__(self, header):
        self.header, self.count = header, 0

    def __call__(self, pipe, mode, spec_path, log):
        assert mode == "build"
        self.count += 1
        spec = json.loads(Path(spec_path).read_text())
        b = Path(spec["build_dir"])
        b.mkdir(parents=True, exist_ok=True)
        (b / spec["toplevel"]).write_text("exe")
        src = spec["sources"][0]
        (b / "Vtop__ver.d").write_text(
            f"{b}/Vtop.cpp {b}/Vtop__ver.d  : /usr/bin/verilator_bin {src} {self.header} \\\n"
            f" /usr/bin/verilator_bin\n")
        return 0


def build_pipeline(tmp_path, fake, src):
    cfg = config(tmp_path, fake)
    cfg.sim = runner.SimSpec(sources=[str(src)], toplevel="top")
    pipe = runner.Pipeline(cfg)
    pipe.out.mkdir(parents=True, exist_ok=True)
    return pipe


@pytest.fixture
def fake_tools(monkeypatch):
    monkeypatch.setattr(runner, "tool_versions", lambda: ("Verilator 5", "2.1.0"))


def test_the_build_is_reused_until_an_included_file_changes(tmp_path, fake_tvla, fake_tools,
                                                            monkeypatch):
    src, header = tmp_path / "top.sv", tmp_path / "defs.svh"
    src.write_text('`include "defs.svh"\nmodule top; endmodule\n')
    header.write_text("`define W 8\n")
    worker = FakeBuild(header)
    monkeypatch.setattr(runner.Pipeline, "_run_worker",
                        lambda self, mode, spec, log: worker(self, mode, spec, log))
    build_pipeline(tmp_path, fake_tvla, src)._build()
    build_pipeline(tmp_path, fake_tvla, src)._build()
    assert worker.count == 1  # reused
    header.write_text("`define W 16\n")  # only the header changes
    build_pipeline(tmp_path, fake_tvla, src)._build()
    assert worker.count == 2
    record = json.loads((tmp_path / "out" / "build" / "scasim-tvla-build.json").read_text())
    assert str(header) in record["inputs"]


def test_a_build_without_a_dependency_file_is_not_reused(tmp_path, fake_tvla, fake_tools,
                                                         monkeypatch):
    src = tmp_path / "top.sv"
    src.write_text("module top; endmodule\n")
    count = []

    def worker(self, mode, spec_path, log):
        spec = json.loads(Path(spec_path).read_text())
        Path(spec["build_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(spec["build_dir"]) / "top").write_text("exe")
        count.append(1)
        return 0

    monkeypatch.setattr(runner.Pipeline, "_run_worker", worker)
    build_pipeline(tmp_path, fake_tvla, src)._build()
    build_pipeline(tmp_path, fake_tvla, src)._build()
    assert len(count) == 2


def test_parse_dep_file_reads_the_prerequisites():
    text = "/b/Vtop.cpp /b/Vtop.h  : /v/bin /src/my\\ file.sv \\\n /inc/a.svh /v/bin\n"
    assert runner.parse_dep_file(text) == ["/v/bin", "/src/my file.sv", "/inc/a.svh", "/v/bin"]
