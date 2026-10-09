"""scasim-tvla: TVLA harness support for cocotb testbenches.

`scasim_tvla.meta` needs only the standard library. `Tvla` and `tvla_test` need cocotb 2.1,
so they are imported on first use.
"""

__version__ = "0.1.0"

__all__ = ["Tvla", "tvla_test", "MetaWriter"]


def __getattr__(name):
    if name == "Tvla":
        from .session import Tvla

        return Tvla
    if name == "tvla_test":
        from .test import tvla_test

        return tvla_test
    if name == "MetaWriter":
        from .meta import MetaWriter

        return MetaWriter
    raise AttributeError(f"module 'scasim_tvla' has no attribute {name!r}")
