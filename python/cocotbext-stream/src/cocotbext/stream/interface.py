"""`StreamInterface`: the signals of one valid/ready stream, resolved once."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

__all__ = ["StreamInterface"]

_ARRAY = re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*$")
_AUTO = "auto"


def _join(parts: Sequence[str | None], separator: str) -> str:
    return separator.join(p for p in parts if p)


def _resolve(entity: Any, name: str) -> Any:
    try:
        return getattr(entity, name)
    except AttributeError:
        raise AttributeError(f"stream signal '{name}' not found in '{entity._name}'") from None


def _to_int(value: Any) -> Any:
    try:
        return int(value)
    except ValueError:
        return str(value)  # the value has x or z bits: keep the bit string for the report


class StreamInterface:
    """The handles of one valid/ready stream.

    All handles are resolved here, once. The cycle loops of the driver and the monitor do no
    lookups.

    Signal names:

    - Derived names: `prefix`, `separator`, and `data_prefix` build the names. `valid` is
      `{prefix}{sep}valid`, `ready` is `{prefix}{sep}ready`, and a data field `f` is
      `{prefix}{sep}{data_prefix}{sep}f`. Empty parts are left out.
    - Explicit names: `valid=`, `ready=`, and a `{field: hdl_name}` mapping in `fields` are used
      as given, without a prefix.

    Args:
        entity: the handle that holds the signals (the DUT or a sub-block).
        prefix: the common name prefix, or None.
        separator: the string between name parts.
        data_prefix: a part between `prefix` and the field name (for example `bits`).
        fields: the data fields. A list of names, or a `{field: hdl_name}` mapping.
            A name of the form `"vec[0:3]"` is a flattened array: its elements are the signals
            `vec_0` to `vec_3` (the separator is used), and its value is a list of four values.
            Element `j` of the list is the signal with index `first + j` (or `first - j` when
            the range counts down). A packed value is an ordinary wide signal and its value is an
            int. Without `fields`, `data_prefix` (if given) names one packed field called `data`.
        valid: explicit name of the valid signal.
        ready: `"auto"` (the default) uses the ready signal if the entity has one. `None` means
            the stream has no ready signal. A string is an explicit name and must exist.
    """

    def __init__(
        self,
        entity: Any,
        prefix: str | None = None,
        *,
        separator: str = "_",
        data_prefix: str | None = None,
        fields: Sequence[str] | Mapping[str, str] | None = None,
        valid: str | None = None,
        ready: str | None = _AUTO,
    ):
        self.entity = entity
        self.prefix = prefix
        self.separator = separator
        self.name = prefix or getattr(entity, "_name", "stream")

        self.valid = _resolve(entity, valid or _join([prefix, "valid"], separator))
        if ready is None:
            self.ready = None
        elif ready == _AUTO:
            ready_name = _join([prefix, "ready"], separator)
            self.ready = getattr(entity, ready_name, None)
        else:
            self.ready = _resolve(entity, ready)

        if fields is None:
            pairs = [("data", _join([prefix, data_prefix], separator))] if data_prefix else []
        elif isinstance(fields, Mapping):
            pairs = list(fields.items())
        else:
            pairs = [(f, _join([prefix, data_prefix, f.split("[")[0].strip()], separator)) for f in fields]

        # One plan entry per field: (name, scalar handle or None, array handles or None).
        plan: list[tuple[str, Any, tuple[Any, ...] | None]] = []
        for spec, hdl_name in pairs:
            match = _ARRAY.match(spec)
            if match is None:
                plan.append((spec, _resolve(entity, hdl_name), None))
                continue
            field, first, last = match.group(1), int(match.group(2)), int(match.group(3))
            step = 1 if last >= first else -1
            handles = tuple(
                _resolve(entity, f"{hdl_name}{separator}{i}") for i in range(first, last + step, step)
            )
            plan.append((field, None, handles))
        self._plan = tuple(plan)
        self.fields = tuple(name for name, _, _ in plan)
        self._scalars = {name: h for name, h, hs in plan if hs is None}
        self._arrays = {name: hs for name, h, hs in plan if hs is not None}

    @property
    def has_ready(self) -> bool:
        return self.ready is not None

    def write(self, tx: Mapping[str, Any]) -> None:
        """Drive the data fields in `tx`. Fields that are missing or None keep their value."""
        scalars = self._scalars
        for name, value in tx.items():
            if value is None:
                continue
            handle = scalars.get(name)
            if handle is not None:
                handle.value = value
                continue
            handles = self._arrays.get(name)
            if handles is None:
                raise KeyError(f"'{name}' is not a field of stream '{self.name}' (fields: {self.fields})")
            if len(value) != len(handles):
                raise ValueError(f"field '{name}' of '{self.name}' has {len(handles)} elements, got {len(value)}")
            for handle, element in zip(handles, value):
                handle.value = element

    def read(self) -> dict[str, Any]:
        """Return the current value of every data field: an int, or a list of ints for an array.

        A value with x or z bits is returned as its bit string.
        """
        out: dict[str, Any] = {}
        for name, handle, handles in self._plan:
            if handles is None:
                out[name] = _to_int(handle.value)
            else:
                out[name] = [_to_int(h.value) for h in handles]
        return out
