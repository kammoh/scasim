"""Random streams and class schedules (standard library only).

Each stream is a `random.Random` seeded from a hash of (base seed, batch id, stream name).
The hash is SHA-256, not Python's `hash()`, so the seeds are the same in every process.
"""

from __future__ import annotations

import hashlib
import math
import random
from fractions import Fraction
from typing import Mapping

STREAMS = ("schedule", "stimulus", "idle", "design", "warmup")
MAX_BLOCK = 10_000


def derive_seed(base: int, batch: str, name: str) -> int:
    """A 64-bit seed for one named stream of one batch."""
    text = f"scasim-tvla/1\0{int(base)}\0{batch}\0{name}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def derived_seeds(base: int, batch: str) -> dict[str, int]:
    return {name: derive_seed(base, batch, name) for name in STREAMS}


def make_streams(base: int, batch: str) -> dict[str, random.Random]:
    return {name: random.Random(seed) for name, seed in derived_seeds(base, batch).items()}


def _fraction(weight) -> Fraction:
    if isinstance(weight, float):
        if not math.isfinite(weight):
            raise ValueError(f"weight {weight!r} is not finite")
        return Fraction(repr(weight))  # 0.1 means 1/10, not the binary value
    return Fraction(weight)


def block_counts(weights: Mapping[int, float]) -> dict[int, int]:
    """The smallest integer counts per label that have the ratios of `weights`."""
    if not weights:
        raise ValueError("no weights")
    fr = {label: _fraction(w) for label, w in weights.items()}
    if any(f <= 0 for f in fr.values()):
        raise ValueError("weights must be positive")
    lcm = math.lcm(*(f.denominator for f in fr.values()))
    counts = {label: int(f * lcm) for label, f in fr.items()}
    gcd = math.gcd(*counts.values())
    counts = {label: c // gcd for label, c in counts.items()}
    if sum(counts.values()) > MAX_BLOCK:
        raise ValueError(
            f"the weights need a block of {sum(counts.values())} segments (limit {MAX_BLOCK}); "
            "use simpler weights or the iid schedule"
        )
    return counts


def make_schedule(
    rng: random.Random, weights: Mapping[int, float], n: int, kind: str = "iid"
) -> list[int]:
    """The labels of `n` segments. Uses only `rng`."""
    labels = sorted(weights)
    if kind == "iid":
        block_counts(weights)  # validates the weights
        return rng.choices(labels, weights=[float(_fraction(weights[k])) for k in labels], k=n)
    if kind == "blocks":
        counts = block_counts(weights)
        block = [label for label in labels for _ in range(counts[label])]
        full, rest = divmod(n, len(block))
        out: list[int] = []
        for _ in range(full):
            rng.shuffle(block)
            out.extend(block)
        total = sum(counts.values())
        if rest:  # the remainder is drawn iid with the same weights
            out.extend(rng.choices(labels, weights=[counts[k] / total for k in labels], k=rest))
        return out
    raise ValueError(f"unknown schedule {kind!r}; use 'iid' or 'blocks'")
