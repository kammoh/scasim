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
