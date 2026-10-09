"""Idle policies: how many idle cycles come before each transaction."""

from __future__ import annotations

import math
import random

__all__ = ["IdlePolicy"]


class IdlePolicy:
    """Number of idle cycles before each transaction.

    Build one with `IdlePolicy.none()`, `IdlePolicy.fixed(n)`, or `IdlePolicy.random(p)`.
    The policy holds no random state. The caller passes its own `random.Random`, so each
    driver has its own stream and the global `random` module is never touched.
    """

    __slots__ = ("_kind", "_n", "_p", "_log_p", "_max_idle")

    def __init__(self, kind: str, n: int = 0, p: float = 0.0, max_idle: int | None = None):
        self._kind = kind
        self._n = n
        self._p = p
        self._log_p = math.log(p) if p > 0.0 else 0.0
        self._max_idle = max_idle

    @classmethod
    def none(cls) -> IdlePolicy:
        """No idle cycles: transactions go back to back."""
        return cls("none")

    @classmethod
    def fixed(cls, n: int) -> IdlePolicy:
        """Exactly `n` idle cycles before each transaction."""
        if n < 0:
            raise ValueError(f"idle cycles must not be negative, got {n}")
        return cls("none") if n == 0 else cls("fixed", n=n)

    @classmethod
    def random(cls, p: float, max_idle: int | None = None) -> IdlePolicy:
        """Each cycle before a transaction is idle with probability `p` (0 <= p < 1).

        The idle count is geometric. `max_idle` caps it. One draw of the random stream
        gives one count.
        """
        if not 0.0 <= p < 1.0:
            raise ValueError(f"probability must be in [0, 1), got {p}")
        if max_idle is not None and max_idle < 0:
            raise ValueError(f"max_idle must not be negative, got {max_idle}")
        if p == 0.0 or max_idle == 0:
            return cls("none")
        return cls("random", p=p, max_idle=max_idle)

    @property
    def is_none(self) -> bool:
        return self._kind == "none"

    def draw(self, rng: random.Random) -> int:
        """Return the number of idle cycles for the next transaction."""
        kind = self._kind
        if kind == "none":
            return 0
        if kind == "fixed":
            return self._n
        # P(count >= k) = p**k, so count = floor(log(u) / log(p)) for u in (0, 1].
        count = int(math.log(1.0 - rng.random()) / self._log_p)
        cap = self._max_idle
        return count if cap is None or count < cap else cap
