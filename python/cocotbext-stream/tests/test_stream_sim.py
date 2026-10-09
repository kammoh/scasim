"""Run tests/tb/tb_stream.py in Verilator and report each cocotb test as a pytest case."""

import pytest

CASES = [
    "registered_ready",
    "fifo_drains_last_item_at_the_accepting_edge",
    "back_to_back",
    "stalls_hold_data_stable",
    "no_ready_signal",
    "random_idle_and_backpressure",
    "explicit_signal_map",
    "interface_errors",
    "scoreboard_mismatch_report",
    "out_of_order_matching",
    "missing_item_at_completion",
    "duplicate_item_at_completion",
]


@pytest.mark.parametrize("name", CASES)
def test_cocotb_case(stream_sim, name):
    assert name in stream_sim.results, f"cocotb did not run {name}: {sorted(stream_sim.results)}"
    assert stream_sim.results[name] is None, stream_sim.results[name]


@pytest.fixture(scope="session")
def failfast_sim(tmp_path_factory):
    from conftest import sim_available, run_cocotb

    ok, why = sim_available()
    if not ok:
        pytest.skip(why)
    return run_cocotb(tmp_path_factory.mktemp("failfast_sim"), "tb_failfast")


def test_fail_fast_mismatch_fails_the_cocotb_test(failfast_sim):
    text = failfast_sim.results["fail_fast_mismatch"]
    assert text is not None, "a mismatch with fail_fast must fail the cocotb test"
    assert "data: expected 0x1235, got 0x1234" in text
    assert "missing" not in text


def test_fail_fast_unexpected_item_fails_the_cocotb_test(failfast_sim):
    text = failfast_sim.results["fail_fast_unexpected"]
    assert text is not None, "an unexpected item with fail_fast must fail the cocotb test"
    assert "unexpected item" in text
