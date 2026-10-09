"""Unit tests of the runner helpers. No simulator and no tvla binary needed."""

import os

import pytest

from scasim_tvla import _args, runner


# -- the tvla argument partition ----------------------------------------------------------


def test_partition_splits_preprocessing_from_analysis():
    p = _args.partition(
        ["--clock", "top.clk", "--include=scope:top", "--exclude", "signal:top.clk",
         "-d", "3", "--pair", "1", "2", "--pool-groups", "--plot=false", "--num-threads", "2"]
    )
    assert p.preprocess == ["--clock", "top.clk", "--include", "scope:top",
                            "--exclude", "signal:top.clk"]
    assert p.select == ["--include", "scope:top", "--exclude", "signal:top.clk"]
    assert p.merge == ["-d", "3", "--pair", "1", "2", "--pool-groups", "--plot=false",
                       "--num-threads", "2"]
    # The per-batch run gets the group choice and the thread count, but no analysis flags.
    assert p.batch == ["--clock", "top.clk", "--include", "scope:top",
                       "--exclude", "signal:top.clk", "--pool-groups", "--num-threads", "2"]
    assert p.clock == "top.clk"


def test_partition_handles_short_option_forms():
    assert _args.partition(["--clock=a.clk", "-d2"]).merge == ["-d", "2"]
    assert _args.partition(["--clock=a.clk", "-d", "4"]).merge == ["-d", "4"]


def test_partition_keeps_rule_order():
    p = _args.partition(["--clock", "c", "--exclude", "scope:a", "--include", "signal:a.x"])
    assert p.select == ["--exclude", "scope:a", "--include", "signal:a.x"]


def test_partition_drops_options_that_the_cache_decides():
    p = _args.partition(["--clock", "c", "--length-policy", "pad", "--use-existing=false"])
    assert "--length-policy" in p.preprocess
    assert "--length-policy" not in p.merge
    assert not any("use-existing" in a for a in p.preprocess + p.merge + p.batch)


@pytest.mark.parametrize(
    "bad",
    [["--clock", "c", "--curve", "final"], ["--clock", "c", "--ttest-output-dir", "x"],
     ["--clock", "c", "--stats-out", "x"], ["--clock", "c", "--meta-json", "x"],
     ["--clock", "c", "--merge-stats", "x"], ["--clock", "c", "--list-signals"],
     ["--clock", "c", "--bogus"]],
)
def test_partition_rejects_options_that_the_runner_owns_or_that_do_not_exist(bad):
    with pytest.raises(_args.ArgsError):
        _args.partition(bad)


def test_partition_needs_a_value():
    with pytest.raises(_args.ArgsError):
        _args.partition(["--clock"])


def test_partition_requires_a_clock_for_version_1_metadata():
    with pytest.raises(_args.ArgsError, match="--clock"):
        _args.partition(["--include", "scope:a"], need_clock=True)
    assert _args.partition([], need_clock=False).clock is None


def test_curve_values():
    for ok in ("every", "every:5", "final"):
        assert _args.check_curve(ok) == ok
    for bad in ("every:0", "every:x", "some", ""):
        with pytest.raises(_args.ArgsError):
            _args.check_curve(bad)


# -- environment, ids, seeds ---------------------------------------------------------------


def test_clean_env_removes_what_the_runner_sets_and_keeps_the_rest():
    env = {
        "PATH": "/bin", "COCOTB_RANDOM_SEED": "1", "COCOTB_TEST_FILTER": "x",
        "SCASIM_TVLA_SEED": "2", "SCASIM_TVLA_BIN": "/x", "WAVES": "1", "GUI": "1",
        "TOPLEVEL": "t", "TOPLEVEL_LANG": "verilog", "MODULE": "m", "TESTCASE": "c",
        "GPI_EXTRA": "keep", "MY_VAR": "keep",
    }
    assert runner.clean_env(env) == {"PATH": "/bin", "GPI_EXTRA": "keep", "MY_VAR": "keep"}


def test_batch_ids_are_zero_padded_in_sorted_order():
    assert runner.batch_ids(3) == ["b0000", "b0001", "b0002"]
    ids = runner.batch_ids(12345)
    assert ids[0] == "b00000" and ids[-1] == "b12344" and ids == sorted(ids)


def test_batch_seed_is_stable_and_depends_on_base_and_batch():
    a = runner.batch_seed(7, "b0001")
    assert a == runner.batch_seed(7, "b0001")
    assert a != runner.batch_seed(7, "b0002") and a != runner.batch_seed(8, "b0001")
    assert 0 <= a < 2**32


# -- trace scope rules ---------------------------------------------------------------------


def test_vlt_text_uses_the_verilator_syntax_without_a_top_prefix():
    text = runner.vlt_text("dut_top", ["dut_top.u_core"], ["dut_top.u_core.u_rng"])
    assert text == (
        "`verilator_config\n"
        'tracing_off -scope "dut_top"\n'
        'tracing_on -scope "dut_top.u_core"\n'
        'tracing_off -scope "dut_top.u_core.u_rng"\n'
    )


def test_vlt_text_is_none_without_rules():
    assert runner.vlt_text("t", [], []) is None


def test_vlt_text_with_only_off_rules_keeps_everything_else():
    assert runner.vlt_text("t", [], ["t.a*"]) == '`verilator_config\ntracing_off -scope "t.a*"\n'


@pytest.mark.parametrize("bad", ["TOP.t.x", 't."x', "", "t.x\n"])
def test_vlt_text_rejects_bad_scope_names(bad):
    with pytest.raises(ValueError):
        runner.vlt_text("t", [bad], [])
    with pytest.raises(ValueError):
        runner.vlt_text("t", [], [bad])


# -- sources and the build key -------------------------------------------------------------


def test_parse_sources_reads_file_lists(tmp_path):
    (tmp_path / "a.sv").write_text("module a; endmodule\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.sv").write_text("module b; endmodule\n")
    (tmp_path / "files.f").write_text(
        "# comment\n\na.sv\nsub/b.sv\n+incdir+inc\n-Wno-fatal -DX=1\n"
    )
    files, args = runner.parse_sources([str(tmp_path / "files.f")])
    assert files == [tmp_path / "a.sv", tmp_path / "sub" / "b.sv"]
    assert args == ["+incdir+inc", "-Wno-fatal", "-DX=1"]


def test_parse_sources_accepts_plain_files_and_rejects_missing_ones(tmp_path):
    (tmp_path / "a.sv").write_text("x")
    files, args = runner.parse_sources([str(tmp_path / "a.sv")])
    assert files == [tmp_path / "a.sv"] and args == []
    with pytest.raises(FileNotFoundError):
        runner.parse_sources([str(tmp_path / "missing.sv")])
    (tmp_path / "bad.f").write_text("nope.sv\n")
    with pytest.raises(FileNotFoundError):
        runner.parse_sources([str(tmp_path / "bad.f")])


def test_build_key_depends_on_everything_that_changes_the_build(tmp_path):
    src = tmp_path / "a.sv"
    src.write_text("module a; endmodule\n")

    def key(**kw):
        base = dict(sources=[src], toplevel="a", build_args=["-O3"], vlt=None, verilator="V 5",
                    cocotb="2.1.0")
        base.update(kw)
        return runner.build_key(**base)

    k = key()
    assert k == key()
    assert k != key(toplevel="b")
    assert k != key(build_args=["-O3", "-GX=1"])
    assert k != key(vlt="`verilator_config\n")
    assert k != key(verilator="V 6")
    assert k != key(cocotb="2.2.0")
    src.write_text("module a; wire x; endmodule\n")
    assert k != key()


def test_default_build_args_are_o3_and_fst_only(tmp_path):
    args = runner.verilator_args(["-GLEAK=0"], trace_depth=None)
    assert args == ["-O3", "--trace-fst", "-GLEAK=0"]
    assert "--trace-threads" not in args and "--x-initial" not in args
    assert runner.verilator_args([], trace_depth=2)[-2:] == ["--trace-depth", "2"]


# -- jobs, tvla binary, signal check ---------------------------------------------------------


def test_plan_jobs_uses_cores_ram_and_disk():
    gb = 10**9
    assert runner.plan_jobs(None, cores=8, free_disk=100 * gb, total_ram=64 * gb) == 8
    assert runner.plan_jobs(None, cores=8, free_disk=3 * gb, total_ram=64 * gb) == 3  # 3 / (2 * 0.5)
    assert runner.plan_jobs(None, cores=8, free_disk=100 * gb, total_ram=4 * gb) == 2
    assert runner.plan_jobs(None, cores=8, free_disk=0, total_ram=64 * gb) == 1
    assert runner.plan_jobs(5, cores=8, free_disk=0, total_ram=0) == 5  # the user decides
    # A real waveform replaces the placeholder: 4 GB free and 1 GB per waveform give 2 jobs.
    assert runner.plan_jobs(None, cores=8, free_disk=4 * gb, total_ram=64 * gb,
                            largest_seen=gb) == 2
    with pytest.raises(ValueError):
        runner.plan_jobs(0, cores=8, free_disk=gb, total_ram=gb)


def test_disk_estimate_uses_the_largest_waveform_seen():
    assert runner.waveform_estimate(0) == 500_000_000
    assert runner.waveform_estimate(1_000) == 1_000  # the largest seen replaces it
    assert runner.waveform_estimate(2_000_000_000) == 2_000_000_000


def test_find_tvla_order(tmp_path, monkeypatch):
    a, b, c = (tmp_path / n for n in "abc")
    for p in (a, b, c):
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
    monkeypatch.setenv("SCASIM_TVLA_BIN", str(b))
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}/nonexistent")
    assert runner.find_tvla(str(a)) == a
    assert runner.find_tvla(None) == b
    monkeypatch.delenv("SCASIM_TVLA_BIN")
    c2 = tmp_path / "tvla"
    c2.write_text("#!/bin/sh\n")
    c2.chmod(0o755)
    assert runner.find_tvla(None) == c2
    c2.unlink()
    with pytest.raises(runner.RunnerError, match="SCASIM_TVLA_BIN"):
        runner.find_tvla(None)
    with pytest.raises(runner.RunnerError):
        runner.find_tvla(str(tmp_path / "missing"))


def test_check_signals_accepts_a_good_selection():
    out = "0\tno\t$rootio.clk\n5\tyes\ttiny.clk\n6\tyes\ttiny.q\n"
    err = "2 of 3 selectable signals are selected\n"
    names = runner.check_signals(out, err, clock="tiny.clk")
    assert names == 2


def test_check_signals_fails_on_warnings_and_empty_selections():
    out = "0\tno\ttiny.clk\n"
    with pytest.raises(runner.RunnerError, match="no signal is selected"):
        runner.check_signals(out, "0 of 1 selectable signals are selected\n", clock=None)
    out = "5\tyes\ttiny.clk\n"
    with pytest.raises(runner.RunnerError, match="matches no signal"):
        runner.check_signals(
            out, "1 of 1 selectable signals are selected\nwarning: the rule +signal:x matches no signal\n",
            clock=None)


def test_check_signals_checks_that_the_clock_is_listed():
    out = "5\tyes\ttiny.clk, $rootio.clk (alias)\n6\tyes\ttiny.q\n"
    err = "2 of 2 selectable signals are selected\n"
    assert runner.check_signals(out, err, clock="$rootio.clk") == 2
    with pytest.raises(runner.RunnerError, match="clock"):
        runner.check_signals(out, err, clock="tiny.clock")
