"""Low-overhead valid/ready stream drivers, monitors, and scoreboards for cocotb 2.1."""

from .driver import ReadyDriver, StreamDriver
from .helpers import reset, start_clock
from .idle import IdlePolicy
from .interface import StreamInterface
from .monitor import StreamMonitor
from .scoreboard import Channel, FieldDiff, Mismatch, Scoreboard

__version__ = "0.1.0"
__all__ = [
    "Channel",
    "FieldDiff",
    "IdlePolicy",
    "Mismatch",
    "ReadyDriver",
    "Scoreboard",
    "StreamDriver",
    "StreamInterface",
    "StreamMonitor",
    "reset",
    "start_clock",
]
