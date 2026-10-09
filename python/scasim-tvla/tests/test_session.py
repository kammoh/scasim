"""Session tests that need no simulator. The clock is faked."""

import asyncio
import random

import pytest

pytest.importorskip("cocotb")

from scasim_tvla import session  # noqa: E402
from scasim_tvla._schedule import make_schedule, make_streams  # noqa: E402
from scasim_tvla.meta import read_meta  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(session, "_now", c)
    monkeypatch.setattr(session, "_time_precision", lambda: -12)
    for k in ("SCASIM_TVLA_SEED", "SCASIM_TVLA_BATCH", "SCASIM_TVLA_OUT", "SCASIM_TVLA_TESTS",
              "SCASIM_TVLA_DESIGN_RANDOM", "SCASIM_TVLA_WAVEFORM"):
        monkeypatch.delenv(k, raising=False)
    return c


def run(coro):
    return asyncio.run(coro)


def drive(tvla, clock, n, body=None, step=100):
    async def main():
        got = []
        async for seg in tvla.segments(n):
            got.append((seg.id, seg.label, seg.warmup, seg.input))
            if body:
                await body(seg)
            clock.t += step
        return got

    return run(main())


def test_commit_and_stable_ids(tmp_path, clock):
    tvla = session.Tvla(None, {0: 5, 1: 6}, warmup=2, seed=7, batch="b3", out=tmp_path)
    got = drive(tvla, clock, 4)
    meta = read_meta(tmp_path / "meta.json")
    assert meta["batch"]["status"] == "committed"
    assert meta["batch"]["id"] == "b3"
    assert [s["id"] for s in meta["segments"]] == [2, 3, 4, 5]
    assert [g[0] for g in got] == [0, 1, 2, 3, 4, 5]
    assert [g[2] for g in got] == [True, True, False, False, False, False]
    # segments are adjacent in time and the warm-up ones are not listed
    assert meta["segments"][0]["start"] == 1200
    assert meta["segments"][0]["end"] == 1300
    assert meta["segments"][1]["start"] == 1300
    assert meta["time"] == {"mantissa": 1, "exponent": -12}
    assert set(meta["batch"]["seeds"]) >= {"base", "schedule", "stimulus", "idle", "design"}
    assert meta["batch"]["seeds"]["base"] == 7


def test_schedule_matches_the_stream(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=1, seed=11, batch="bx", out=tmp_path)
    drive(tvla, clock, 50)
    labels = [s["label"] for s in read_meta(tmp_path / "meta.json")["segments"]]
    expect = make_schedule(make_streams(11, "bx")["schedule"], {0: 1, 1: 1}, 50, "iid")
    assert labels == expect


def test_warmup_does_not_change_schedule_or_inputs(tmp_path, clock):
    def go(warmup, sub):
        clock.t = 1000
        tvla = session.Tvla(None, {0: 1, 1: lambda rng: rng.getrandbits(32)}, warmup=warmup,
                            seed=5, batch="b", out=tmp_path / sub)
        got = drive(tvla, clock, 20)
        return [g for g in got if not g[2]]

    a, b = go(0, "a"), go(3, "b")
    assert [g[1] for g in a] == [g[1] for g in b]
    assert [g[3] for g in a] == [g[3] for g in b]


def test_blocks_schedule(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, weights={0: 1, 1: 3}, warmup=0, schedule="blocks", seed=1, batch="b", out=tmp_path)
    drive(tvla, clock, 8)
    labels = [s["label"] for s in read_meta(tmp_path / "meta.json")["segments"]]
    assert sorted(labels[:4]) == [0, 1, 1, 1]
    assert sorted(labels[4:]) == [0, 1, 1, 1]


def test_env_variables(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("SCASIM_TVLA_SEED", "99")
    monkeypatch.setenv("SCASIM_TVLA_BATCH", "b0042")
    monkeypatch.setenv("SCASIM_TVLA_OUT", str(tmp_path / "o"))
    monkeypatch.setenv("SCASIM_TVLA_WAVEFORM", "x.fst")
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0)
    drive(tvla, clock, 3)
    meta = read_meta(tmp_path / "o" / "meta.json")
    assert meta["batch"]["id"] == "b0042"
    assert meta["batch"]["seeds"]["base"] == 99
    assert meta["waveform"] == "x.fst"


def test_env_test_count_overrides(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("SCASIM_TVLA_TESTS", "5")
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, out=tmp_path)
    drive(tvla, clock, 100)
    assert len(read_meta(tmp_path / "meta.json")["segments"]) == 5


def test_same_seed_same_result_different_batch_differs(tmp_path, clock):
    def go(batch, sub):
        clock.t = 1000
        tvla = session.Tvla(None, {0: 1, 1: lambda rng: rng.getrandbits(32)}, warmup=1,
                            seed=3, batch=batch, out=tmp_path / sub)
        return drive(tvla, clock, 30)

    assert go("b1", "a") == go("b1", "b")
    assert go("b1", "c") != go("b2", "d")


def test_input_forms(tmp_path, clock):
    seen = {}

    def with_rng(rng):
        seen["rng"] = rng
        return 1

    def legacy():
        return random.getrandbits(32)

    def optional(n=16):
        seen["n"] = n
        return n

    tvla = session.Tvla(None, {0: with_rng, 1: legacy, 2: optional, 3: 77}, warmup=0, seed=1, batch="b",
                        out=tmp_path)

    async def main():
        async for seg in tvla.segments(60):
            v = seg.input
            assert v == seg.input  # cached per segment
            if seg.label == 0:
                assert v == 1 and seen["rng"] is seg.rng
            if seg.label == 2:
                assert v == 16 and seen["n"] == 16
            if seg.label == 3:
                assert v == 77
            clock_advance()

    def clock_advance():
        clock.t += 10

    run(main())


def test_legacy_generators_use_seeded_global_random(tmp_path, clock):
    def go(sub):
        clock.t = 1000
        tvla = session.Tvla(None, {0: 1, 1: lambda: random.getrandbits(32)}, warmup=0,
                            seed=8, batch="b", out=tmp_path / sub)
        random.seed(12345 if sub == "a" else 999)  # whatever the user did before
        return drive(tvla, clock, 30)

    assert go("a") == go("b")


def test_failure_gives_diagnostic_with_context_manager(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=1, seed=1, batch="b", out=tmp_path)

    async def main():
        with tvla:
            async for seg in tvla.segments(5):
                if seg.id == 3:
                    raise AssertionError("boom")
                clock.t += 10

    with pytest.raises(AssertionError):
        run(main())
    meta = read_meta(tmp_path / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert [s["id"] for s in meta["segments"]] == [1, 2]  # the open segment 3 is not listed


def test_pass_with_context_manager_commits(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        with tvla:
            async for seg in tvla.segments(3):
                clock.t += 10

    run(main())
    assert read_meta(tmp_path / "meta.json")["batch"]["status"] == "committed"


def test_failure_after_loop_is_diagnostic_with_context_manager(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        with tvla:
            async for seg in tvla.segments(3):
                clock.t += 10
            raise AssertionError("final check")

    with pytest.raises(AssertionError):
        run(main())
    assert read_meta(tmp_path / "meta.json")["batch"]["status"] == "diagnostic"


def test_bare_loop_commits_on_completion(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, seed=1, batch="b", out=tmp_path)
    drive(tvla, clock, 3)
    assert read_meta(tmp_path / "meta.json")["batch"]["status"] == "committed"


def test_bare_loop_early_exit_is_diagnostic(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        gen = tvla.segments(5)
        async for seg in gen:
            clock.t += 10
            if seg.id == 1:
                break
        await gen.aclose()

    run(main())
    meta = read_meta(tmp_path / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert [s["id"] for s in meta["segments"]] == [0]


def test_end_at_and_zero_length(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def body(seg):
        clock.t += 50
        seg.end_at(seg.start + 1)

    drive(tvla, clock, 2, body)
    segs = read_meta(tmp_path / "meta.json")["segments"]
    assert [(s["start"], s["end"]) for s in segs] == [(1000, 1001), (1150, 1151)]


def test_end_at_rejects_bad_times(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        async for seg in tvla.segments(1):
            with pytest.raises(ValueError):
                seg.end_at(seg.start)  # not after the start
            with pytest.raises(ValueError):
                seg.end_at(clock.t + 5)  # in the future
            clock.t += 1

    run(main())


def test_zero_length_segment_is_an_error(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        with tvla:
            async for seg in tvla.segments(1):
                pass  # no time passes

    with pytest.raises(ValueError):
        run(main())
    assert read_meta(tmp_path / "meta.json")["batch"]["status"] == "diagnostic"


def test_extend_needs_a_clock(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        async for seg in tvla.segments(1):
            await seg.extend(1)

    with pytest.raises(ValueError):
        run(main())


def test_design_random_requires_hook(tmp_path, clock):
    with pytest.raises(ValueError):
        session.Tvla(None, {0: 1, 1: 1}, design_random_mode="off", out=tmp_path)
    with pytest.raises(ValueError):
        session.Tvla(None, {0: 1, 1: 1}, design_random=lambda r, m: m, design_random_mode="maybe",
                     out=tmp_path)


def test_design_random_hook_contract(tmp_path, clock):
    calls = []

    def rng_seed_const(rng, mode):
        calls.append((rng, mode))
        return "off"

    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=2, batch="b", out=tmp_path,
                        design_random=rng_seed_const, design_random_mode="off")
    drive(tvla, clock, 2)
    assert len(calls) == 1 and calls[0][1] == "off"
    assert calls[0][0] is tvla.design_rng
    dr = read_meta(tmp_path / "meta.json")["batch"]["design_random"]
    assert dr == {"requested": "off", "applied": "off", "how": "hook: rng_seed_const"}


def test_design_random_async_hook_with_dict_result(tmp_path, clock):
    async def hook(rng, mode):
        return {"applied": "on", "how": "custom"}

    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=2, batch="b", out=tmp_path,
                        design_random=hook, design_random_mode="on")
    drive(tvla, clock, 2)
    dr = read_meta(tmp_path / "meta.json")["batch"]["design_random"]
    assert dr == {"requested": "on", "applied": "on", "how": "custom"}


def test_no_design_random_request_is_recorded_as_none(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 1}, warmup=0, seed=2, batch="b", out=tmp_path,
                        design_random=lambda r, m: m)
    drive(tvla, clock, 2)
    meta = read_meta(tmp_path / "meta.json")
    assert meta["batch"]["design_random"]["requested"] == "none"


def test_bad_arguments(tmp_path, clock):
    with pytest.raises(ValueError):
        session.Tvla(None, {}, out=tmp_path)
    with pytest.raises(ValueError):
        session.Tvla(None, {0: 1, 1: 1}, weights={0: 1}, out=tmp_path)
    with pytest.raises(ValueError):
        session.Tvla(None, {0: 1, 1: 1}, schedule="x", out=tmp_path)
    with pytest.raises(ValueError):
        session.Tvla(None, {0: 1, 70000: 1}, out=tmp_path)


def test_break_inside_context_is_a_diagnostic(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=1, seed=1, batch="b", out=tmp_path)

    async def main():
        with tvla:
            async for seg in tvla.segments(5):
                clock.t += 10
                if seg.id == 3:
                    break  # the run is shortened: 2 of 5 scheduled segments were closed

    run(main())
    meta = read_meta(tmp_path / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert [s["id"] for s in meta["segments"]] == [1, 2]
    assert meta["extensions"]["diagnostic"]["reason"] == "schedule not completed: 2 of 5 segments"


def test_context_without_running_the_schedule_is_a_diagnostic(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, seed=1, batch="b", out=tmp_path)
    with tvla:
        pass
    meta = read_meta(tmp_path / "meta.json")
    assert meta["batch"]["status"] == "diagnostic"
    assert meta["extensions"]["diagnostic"]["reason"] == "schedule not completed: 0 of 0 segments"


def test_full_run_has_no_diagnostic_reason(tmp_path, clock):
    tvla = session.Tvla(None, {0: 1, 1: 2}, warmup=0, seed=1, batch="b", out=tmp_path)

    async def main():
        with tvla:
            async for seg in tvla.segments(3):
                clock.t += 10

    run(main())
    assert "diagnostic" not in read_meta(tmp_path / "meta.json")["extensions"]
