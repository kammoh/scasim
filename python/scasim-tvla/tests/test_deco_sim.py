"""Simulator tests for @tvla_test. Fixtures come from conftest.py."""

import json

import pytest

pytest.importorskip("cocotb")

from scasim_tvla.meta import read_meta  # noqa: E402

pytestmark = pytest.mark.usefixtures("sim_available")


def test_decorator_commits_on_pass(run_tb):
    out, fails = run_tb("deco_pass", module="tb_deco")
    assert fails == 0
    meta = read_meta(out / "meta.json")
    assert meta["batch"]["status"] == "committed"
    assert [s["id"] for s in meta["segments"]] == [2, 3, 4, 5, 6]  # two warm-up ids skipped
    assert json.loads((out / "last.json").read_text())["id"] == 6


def test_decorator_diagnostic_on_failure(run_tb):
    out, fails = run_tb("deco_fail", module="tb_deco")
    assert fails == 1
    meta = read_meta(out / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert [s["id"] for s in meta["segments"]] == [1, 2]


def test_decorator_test_count_from_environment(run_tb):
    out, _ = run_tb("deco_pass", tests=2, module="tb_deco")
    assert len(read_meta(out / "meta.json")["segments"]) == 2
