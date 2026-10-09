"""The committed fixture that the Rust tests read (no simulator needed)."""

from pathlib import Path

from scasim_tvla.meta import read_meta

FIXTURE = Path(__file__).parents[3] / "tests" / "data" / "scasim_meta_v1"


def test_committed_fixture_for_rust():
    """Regenerate with `SCASIM_TVLA_WRITE_FIXTURE=1 pytest tests/test_sim.py`."""
    meta = read_meta(FIXTURE / "meta.json")
    assert meta["batch"]["status"] == "committed"
    assert (FIXTURE / meta["waveform"]).stat().st_size < 50_000


