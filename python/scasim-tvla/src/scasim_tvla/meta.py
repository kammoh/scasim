"""Writer for `scasim_meta` version 1 (standard library only).

A `MetaWriter` collects the segments of one batch. It checks each segment when you add it.
`commit()` or `write_diagnostic()` writes the file once, atomically: a temporary file in the
same directory, then a rename. Any Python flow can use it, with or without cocotb.
"""

from __future__ import annotations

import gzip
import json
import operator
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

VERSION = 1
MAX_LABEL = 65535
MAX_SEED = 2**64 - 1

DEFAULT_LABELS = {0: "fixed", 1: "random"}
DEFAULT_GROUPS = {0: "default"}
# Used when nothing is known about the design randomness.
NO_DESIGN_RANDOM = {"requested": "none", "applied": "none", "how": "none"}


class MetaError(ValueError):
    """The metadata is not valid."""


def _int(value: Any, what: str) -> int:
    if isinstance(value, bool):
        raise MetaError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise MetaError(f"{what} must be an integer, got {value!r}") from None


def _names(mapping: Mapping[int, str], what: str, max_id: int) -> dict[int, str]:
    if not mapping:
        raise MetaError(f"{what} must not be empty")
    out = {}
    for key, name in mapping.items():
        key = _int(key, f"{what} id")
        if not 0 <= key <= max_id:
            raise MetaError(f"{what} id {key} is outside 0..={max_id}")
        if not isinstance(name, str):
            raise MetaError(f"{what} name for id {key} must be a string")
        out[key] = name
    return out


class MetaWriter:
    """Collect the segments of one batch and write the `scasim_meta` file.

    `time=(mantissa, exponent)`: one integer time unit lasts `mantissa * 10**exponent` seconds.
    A path that ends in `.gz` is written gzip-compressed.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        time: tuple[int, int] = (1, -12),
        labels: Mapping[int, str] | None = None,
        groups: Mapping[int, str] | None = None,
        seeds: Mapping[str, int] | None = None,
        batch_id: str = "b0000",
        design: Mapping[str, Any] | None = None,
        waveform: str | None = None,
        design_random: Mapping[str, str] | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        mantissa, exponent = time
        mantissa, exponent = _int(mantissa, "time mantissa"), _int(exponent, "time exponent")
        if mantissa < 1:
            raise MetaError("time mantissa must be positive")
        self._time = (mantissa, exponent)
        self._labels = _names(DEFAULT_LABELS if labels is None else labels, "label", MAX_LABEL)
        self._groups = _names(DEFAULT_GROUPS if groups is None else groups, "group", MAX_LABEL)
        self._seeds = {}
        for name, seed in (seeds or {}).items():
            seed = _int(seed, f"seed {name!r}")
            if not 0 <= seed <= MAX_SEED:
                raise MetaError(f"seed {name!r} is outside 0..2**64-1")
            self._seeds[str(name)] = seed
        if not isinstance(batch_id, str):
            raise MetaError("batch id must be a string")
        self._batch_id = batch_id
        self._design = dict(design) if design else None
        self._waveform = waveform
        self._design_random = dict(NO_DESIGN_RANDOM if design_random is None else design_random)
        self._extensions = dict(extensions or {})
        self._segments: list[dict[str, int]] = []
        self._next_id = 0
        self._last_end = 0
        self._done = False

    # -- building -------------------------------------------------------------------------

    def skip_ids(self, count: int = 1) -> None:
        """Skip segment ids (warm-up segments). Only before the first segment."""
        self._check_open()
        count = _int(count, "skip count")
        if count < 0:
            raise MetaError("skip count must not be negative")
        if self._segments:
            raise MetaError("ids can be skipped only before the first segment")
        self._next_id += count

    def segment(self, start: int, end: int, label: int, group: int = 0) -> int:
        """Add one complete segment `[start, end)`. Returns its id."""
        self._check_open()
        start, end = _int(start, "segment start"), _int(end, "segment end")
        label, group = _int(label, "segment label"), _int(group, "segment group")
        if start < 0:
            raise MetaError(f"segment start {start} is negative")
        if end <= start:
            raise MetaError(f"segment end {end} must be greater than its start {start}")
        if start < self._last_end:
            raise MetaError(
                f"segment [{start}, {end}) starts before the end ({self._last_end}) "
                "of the previous segment"
            )
        if label not in self._labels:
            raise MetaError(f"label {label} is not declared")
        if group not in self._groups:
            raise MetaError(f"group {group} is not declared")
        seg_id = self._next_id
        self._segments.append(
            {"id": seg_id, "start": start, "end": end, "label": label, "group": group}
        )
        self._next_id += 1
        self._last_end = end
        return seg_id

    def set_design_random(self, record: Mapping[str, str]) -> None:
        """Set the `design_random` record (requested, applied, how)."""
        self._check_open()
        self._design_random = dict(record)

    # -- writing --------------------------------------------------------------------------

    def commit(self) -> None:
        """Write the file with status `committed`."""
        self._check_open()
        if not self._segments:
            raise MetaError("cannot commit a batch with no segments")
        self._write("committed")

    def write_diagnostic(self) -> None:
        """Write the file with status `diagnostic` (failed or canceled batch)."""
        self._check_open()
        self._write("diagnostic")

    def to_dict(self, status: str) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "scasim_meta": VERSION,
            "batch": {
                "id": self._batch_id,
                "seeds": dict(self._seeds),
                "design_random": dict(self._design_random),
                "status": status,
            },
        }
        if self._design is not None:
            doc["design"] = dict(self._design)
        if self._waveform is not None:
            doc["waveform"] = self._waveform
        doc["time"] = {"mantissa": self._time[0], "exponent": self._time[1]}
        doc["segments"] = [dict(s) for s in self._segments]
        doc["labels"] = {str(k): v for k, v in sorted(self._labels.items())}
        doc["groups"] = {str(k): v for k, v in sorted(self._groups.items())}
        doc["extensions"] = dict(self._extensions)
        return doc

    def _check_open(self) -> None:
        if self._done:
            raise MetaError("the metadata is already written")

    def _write(self, status: str) -> None:
        data = json.dumps(self.to_dict(status), separators=(",", ":")).encode()
        if self.path.name.endswith(".gz"):
            data = gzip.compress(data, mtime=0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        # Created with the normal permissions (the umask applies), unlike tempfile.mkstemp.
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        self._done = True


def read_meta(path: str | os.PathLike) -> dict[str, Any]:
    """Read a `scasim_meta` version 1 file (JSON, or gzip if the path ends in `.gz`)."""
    path = Path(path)
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("scasim_meta") != VERSION:
        raise MetaError(f"{path} is not a scasim_meta version {VERSION} file")
    return doc
