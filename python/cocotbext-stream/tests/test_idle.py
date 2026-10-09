"""IdlePolicy: idle cycles before each transaction, from its own random stream."""

import random

import pytest

from cocotbext.stream import IdlePolicy


def draws(policy, rng, n):
    return [policy.draw(rng) for _ in range(n)]


def test_none_never_idles():
    policy = IdlePolicy.none()
    assert policy.is_none
    assert draws(policy, random.Random(1), 20) == [0] * 20


def test_fixed():
    policy = IdlePolicy.fixed(3)
    assert not policy.is_none
    assert draws(policy, random.Random(1), 5) == [3] * 5


def test_fixed_zero_is_none():
    assert IdlePolicy.fixed(0).is_none


def test_random_is_deterministic_per_stream():
    a = draws(IdlePolicy.random(0.5), random.Random(7), 200)
    b = draws(IdlePolicy.random(0.5), random.Random(7), 200)
    c = draws(IdlePolicy.random(0.5), random.Random(8), 200)
    assert a == b
    assert a != c


def test_random_does_not_touch_the_global_stream():
    random.seed(99)
    expected = random.random()
    random.seed(99)
    draws(IdlePolicy.random(0.5), random.Random(7), 50)
    assert random.random() == expected


def test_random_statistics():
    # Each cycle is idle with probability p, so the idle count is geometric.
    p = 0.6
    values = draws(IdlePolicy.random(p), random.Random(3), 20000)
    mean = sum(values) / len(values)
    assert mean == pytest.approx(p / (1 - p), rel=0.05)
    zero_fraction = values.count(0) / len(values)
    assert zero_fraction == pytest.approx(1 - p, abs=0.02)


def test_random_max_idle_caps():
    values = draws(IdlePolicy.random(0.9, max_idle=4), random.Random(3), 2000)
    assert max(values) == 4


def test_random_zero_probability_is_none():
    assert IdlePolicy.random(0.0).is_none


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_random_rejects_bad_probability(p):
    with pytest.raises(ValueError):
        IdlePolicy.random(p)


def test_fixed_rejects_negative():
    with pytest.raises(ValueError):
        IdlePolicy.fixed(-1)
