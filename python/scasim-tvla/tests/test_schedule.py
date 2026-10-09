import random
from collections import Counter

import pytest

from scasim_tvla._schedule import STREAMS, block_counts, derive_seed, make_schedule, make_streams


def test_derive_seed_is_stable_and_distinct():
    # Fixed values: they must never change between releases or Python processes.
    assert derive_seed(1234, "b0001", "schedule") == derive_seed(1234, "b0001", "schedule")
    seeds = {
        derive_seed(b, bid, s)
        for b in (1, 2)
        for bid in ("b0", "b1")
        for s in STREAMS
    }
    assert len(seeds) == 2 * 2 * len(STREAMS)
    assert all(0 <= s < 2**64 for s in seeds)


def test_derive_seed_has_no_field_confusion():
    assert derive_seed(1, "2", "x") != derive_seed(12, "", "x")
    assert derive_seed(1, "b", "ab") != derive_seed(1, "ba", "b")


def test_derive_seed_pinned_value():
    # Pins the hash construction. If this fails, old results are no longer reproducible.
    assert derive_seed(0, "b0000", "schedule") == 0x92F82C74BBF9DABE


def test_iid_frequencies():
    rng = random.Random(1)
    sched = make_schedule(rng, {0: 1, 1: 3}, 40000, "iid")
    c = Counter(sched)
    assert len(sched) == 40000
    assert abs(c[0] / 40000 - 0.25) < 0.01
    assert abs(c[1] / 40000 - 0.75) < 0.01


def test_iid_is_not_balanced_exactly():
    # iid means independent draws: the counts differ between seeds.
    counts = {Counter(make_schedule(random.Random(s), {0: 1, 1: 1}, 100, "iid"))[0] for s in range(20)}
    assert len(counts) > 3


@pytest.mark.parametrize(
    "weights,expected",
    [
        ({0: 1, 1: 1}, {0: 1, 1: 1}),
        ({0: 1, 1: 3}, {0: 1, 1: 3}),
        ({0: 2, 1: 6}, {0: 1, 1: 3}),
        ({0: 0.5, 1: 0.5}, {0: 1, 1: 1}),
        ({0: 0.25, 1: 0.75}, {0: 1, 1: 3}),
        ({0: 0.1, 1: 0.2, 2: 0.7}, {0: 1, 1: 2, 2: 7}),
        ({0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}),
    ],
)
def test_block_counts_smallest_integers(weights, expected):
    assert block_counts(weights) == expected


def test_block_counts_rejects_bad_weights():
    for bad in ({0: 0, 1: 1}, {0: -1, 1: 1}, {}, {0: float("nan"), 1: 1}):
        with pytest.raises(ValueError):
            block_counts(bad)


def test_block_counts_rejects_huge_blocks():
    with pytest.raises(ValueError):
        block_counts({0: 1, 1: 99991})  # 99992 per block, above the limit


def test_blocks_balance_and_remainder():
    rng = random.Random(7)
    n = 3 * 4 + 3  # three blocks of 4 (1:3), remainder 3
    sched = make_schedule(rng, {0: 1, 1: 3}, n, "blocks")
    assert len(sched) == n
    for i in range(3):
        block = sched[4 * i : 4 * i + 4]
        assert Counter(block) == {0: 1, 1: 3}
    # The remainder is drawn iid, so it may have any counts, but only valid labels.
    assert set(sched[12:]) <= {0, 1}


def test_blocks_are_shuffled():
    sched = make_schedule(random.Random(3), {0: 1, 1: 1}, 2000, "blocks")
    first_positions = Counter(sched[i] for i in range(0, 2000, 2))
    assert 420 < first_positions[0] < 580  # the position of the 0 within a block is random


def test_blocks_remainder_is_iid_not_forced_balanced():
    seen = set()
    for seed in range(50):
        sched = make_schedule(random.Random(seed), {0: 1, 1: 1}, 7, "blocks")
        seen.add(Counter(sched[6:])[0])
    assert seen == {0, 1}  # remainder of 1: both labels occur


def test_unknown_schedule_kind():
    with pytest.raises(ValueError):
        make_schedule(random.Random(0), {0: 1, 1: 1}, 4, "round-robin")


def test_schedule_is_deterministic():
    a = make_schedule(random.Random(5), {0: 1, 1: 1}, 100, "iid")
    b = make_schedule(random.Random(5), {0: 1, 1: 1}, 100, "iid")
    assert a == b
    c = make_schedule(random.Random(6), {0: 1, 1: 1}, 100, "iid")
    assert a != c


def test_streams_are_deterministic_per_batch():
    a = make_streams(99, "b1")
    b = make_streams(99, "b1")
    for name in STREAMS:
        assert [a[name].random() for _ in range(5)] == [b[name].random() for _ in range(5)]
    c = make_streams(99, "b2")
    assert a["stimulus"].random() != c["stimulus"].random()


def test_stream_independence():
    """Draws from one stream never change another stream."""
    ref = make_streams(42, "b0")
    ref_sched = make_schedule(ref["schedule"], {0: 1, 1: 1}, 50, "iid")
    ref_stim = [ref["stimulus"].getrandbits(32) for _ in range(10)]

    s = make_streams(42, "b0")
    for _ in range(12345):  # a lot of draws from the other streams
        s["stimulus"].random()
        s["idle"].random()
        s["design"].random()
        s["warmup"].random()
    assert make_schedule(s["schedule"], {0: 1, 1: 1}, 50, "iid") == ref_sched

    t = make_streams(42, "b0")
    make_schedule(t["schedule"], {0: 1, 1: 1}, 5000, "iid")
    assert [t["stimulus"].getrandbits(32) for _ in range(10)] == ref_stim
