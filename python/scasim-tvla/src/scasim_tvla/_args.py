"""Split the user's `tvla` arguments for the three `tvla` calls of the pipeline.

The runner calls `tvla` in three ways, and each call accepts different options:

- `--list-signals`: the selection rules only.
- `--stats-out` (one call for each batch): all preprocessing options. They decide the traces,
  so they are part of the cache key.
- `--merge-stats` (one call for the report): analysis options only. `tvla` rejects
  preprocessing options there, because the caches already fix them.

The runner owns `--curve`, `--ttest-output-dir`, `--meta-json`, `--meta-list`, `--stats-out`,
`--merge-stats`, `--list-signals`, `--traces-out`, and `--traces-channels` (the runner sets the
two `--traces-` options from its own `--keep traces` and `--traces-channels`). The user must not
pass them after `--`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class ArgsError(ValueError):
    """The `tvla` arguments cannot be used."""


# Options that take one value: name -> number of values.
_PREPROCESS = {
    "--clock": 1, "--edges": 1, "--offset": 1, "--include": 1, "--exclude": 1,
    "--per-scope": 1, "--depth": 1, "--shuffle-labels": 1, "--length-policy": 1,
}
_SELECT = {"--include", "--exclude"}
# Options that the per-batch call also takes: the group choice and the thread count.
_BATCH_ONLY = {"--group": 1, "--pool-groups": 0, "--num-threads": 1}
# Analysis options: the report call takes them.
_MERGE = {
    "-d": 1, "--pair": 2, "--group": 1, "--pool-groups": 0, "--num-threads": 1,
    "--chi2": "optional", "--plot": "optional", "--show": 0,
}
# The cache decides these, or they only matter for the legacy trace cache. Dropped.
_DROP = {"--use-existing": "optional"}
_OWNED = {
    "--curve", "--ttest-output-dir", "--meta-json", "--meta-list", "--stats-out",
    "--merge-stats", "--list-signals", "--traces-out", "--traces-channels",
}

_CURVE = re.compile(r"^(every|final|every:[1-9][0-9]*)$")


@dataclass
class Partition:
    preprocess: list[str] = field(default_factory=list)  # all preprocessing options
    select: list[str] = field(default_factory=list)  # the rules (for --list-signals)
    batch: list[str] = field(default_factory=list)  # for each --stats-out call
    merge: list[str] = field(default_factory=list)  # for the --merge-stats call
    clock: str | None = None
    group_choice: bool = False  # the user gave --group or --pool-groups


def check_curve(value: str) -> str:
    if not _CURVE.match(value):
        raise ArgsError(f"--curve must be every, every:K (K > 0), or final, got {value!r}")
    return value


def _tokens(args: list[str]) -> list[tuple[str, list[str], str]]:
    """Group the tokens into (name, values, form). `form` is 'eq' or 'sep' (optional values)."""
    known = {**_PREPROCESS, **_MERGE, **_BATCH_ONLY, **_DROP}
    out: list[tuple[str, list[str], str]] = []
    i = 0
    while i < len(args):
        tok = args[i]
        i += 1
        name, eq, value = tok.partition("=")
        if re.fullmatch(r"-d[0-9]+", tok):
            out.append(("-d", [tok[2:]], "sep"))
            continue
        if name in _OWNED:
            raise ArgsError(f"{name} is set by scasim-tvla. Do not pass it after --")
        if name not in known:
            raise ArgsError(f"unknown or unsupported tvla option {tok!r}")
        want = known[name]
        if want == "optional":
            out.append((name, [value] if eq else [], "eq"))
            continue
        if eq:
            if want != 1:
                raise ArgsError(f"{name} takes {want} values; use the form {name} V1 V2")
            out.append((name, [value], "sep"))
            continue
        if i + want > len(args):
            raise ArgsError(f"{name} needs {want} value(s)")
        out.append((name, args[i : i + want], "sep"))
        i += want
    return out


def _flat(name: str, values: list[str], form: str) -> list[str]:
    if form == "eq":
        return [f"{name}={values[0]}"] if values else [name]
    return [name, *values]


def partition(args: list[str], need_clock: bool = True) -> Partition:
    """Split `args`. With `need_clock`, require `--clock` (version 1 metadata needs it)."""
    p = Partition()
    for name, values, form in _tokens(list(args)):
        flat = _flat(name, values, form)
        if name in _DROP:
            continue
        if name in _PREPROCESS:
            p.preprocess += flat
            p.batch += flat
            if name in _SELECT:
                p.select += flat
            if name == "--clock":
                p.clock = values[0]
        if name in _BATCH_ONLY and name != "--num-threads":
            p.batch += flat
            p.group_choice = True
        if name == "--num-threads":
            p.batch += flat
        if name in _MERGE:
            p.merge += flat
    if need_clock and p.clock is None:
        raise ArgsError(
            "scasim_meta version 1 needs a clock: pass --clock PATH after --, for example "
            "-- --clock tb.dut.clk (see tvla --list-signals)"
        )
    return p
