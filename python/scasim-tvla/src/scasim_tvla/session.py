"""The TVLA session for cocotb 2.1.

    tvla = Tvla(dut, classes={0: FIXED, 1: gen_random}, warmup=1, clock=dut.clk)
    with tvla:
        async for seg in tvla.segments(num_tests):
            await run_one(dut, seg.input)

A segment opens when the `async for` body starts and closes when the body returns. Its time
range is `[start, end)` in simulator steps (`get_sim_time("step")`). The segment is half-open:
activity at exactly `end` belongs to the next segment. `await RisingEdge(clk)` returns at the
time of that edge, so the last operation edge lands at `end` if the body returns right after it.
**Return after the edge that follows the last operation edge.** Or call `seg.end_at(time)`, or
`await seg.extend(cycles, clock)` at the end of the body.

The session does nothing per cycle. It acts when a segment opens or closes, and once in `extend()`.

Environment variables (set by the runner; the arguments override them):

- `SCASIM_TVLA_SEED`: base seed (default: `cocotb.RANDOM_SEED`)
- `SCASIM_TVLA_BATCH`: batch id (default `b0000`)
- `SCASIM_TVLA_OUT`: output directory of the batch (default: the current directory)
- `SCASIM_TVLA_TESTS`: number of segments per batch. When set, it overrides the `num_tests` value.
- `SCASIM_TVLA_WAVEFORM`: waveform file name, relative to the output directory (default: none)
- `SCASIM_TVLA_DESIGN_RANDOM`: `on` or `off`, the requested design randomness mode (default: none)
"""

from __future__ import annotations

import inspect
import os
import random
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping

import cocotb
import cocotb.simtime as _simtime
from cocotb.triggers import ClockCycles

from ._schedule import derived_seeds, make_schedule, make_streams
from .meta import MAX_LABEL, MetaWriter

ENV_SEED = "SCASIM_TVLA_SEED"
ENV_BATCH = "SCASIM_TVLA_BATCH"
ENV_OUT = "SCASIM_TVLA_OUT"
ENV_TESTS = "SCASIM_TVLA_TESTS"
ENV_WAVEFORM = "SCASIM_TVLA_WAVEFORM"
ENV_DESIGN_RANDOM = "SCASIM_TVLA_DESIGN_RANDOM"


def _now() -> int:
    """The simulator time in steps."""
    return int(_simtime.get_sim_time("step"))


def _time_precision() -> int:
    """The exponent of one simulator step in seconds. Read it inside a running test."""
    return int(_simtime.time_precision)


def _env_int(name: str) -> int | None:
    text = os.environ.get(name)
    if text is None or text == "":
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {text!r}") from None


def _passes_rng(fn: Callable) -> bool:
    """Should `fn` be called with the stimulus rng?

    Yes if it has a parameter named `rng`, a required positional parameter, or `*args`.
    Otherwise it is called without an argument (a legacy generator that uses `random`).
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    P = inspect.Parameter
    for p in params:
        if p.name == "rng" and p.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY):
            return True
    for p in params:
        if p.kind == P.VAR_POSITIONAL:
            return True
        if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD):
            return p.default is P.empty
    return False


class Segment:
    """One trace window. The session creates it."""

    def __init__(self, session: "Tvla", index: int, seg_id: int, label: int, warmup: bool,
                 rng: random.Random) -> None:
        self._session = session
        self.index = index  # position among the scheduled segments (warm-up: its own count)
        self.id = seg_id  # stable metadata id; warm-up ids are skipped in the file
        self.label = label
        self.group = session.group
        self.warmup = warmup
        self.rng = rng  # the stimulus stream (the warm-up stream for warm-up segments)
        self.start = 0
        self._end: int | None = None
        self._has_input = False
        self._input: Any = None

    @property
    def input(self) -> Any:
        """The class value. A generator is called once, with `seg.rng` if it accepts one."""
        if not self._has_input:
            value = self._session._classes[self.label]
            if callable(value):
                value = value(self.rng) if self._session._takes_rng[self.label] else value()
            self._input = value
            self._has_input = True
        return self._input

    def end_at(self, time: int) -> None:
        """Close the segment at `time` (steps) instead of the time the body returns.

        `time` must be after the start and not after the current simulator time.
        """
        time = int(time)
        if time <= self.start:
            raise ValueError(f"end time {time} must be after the segment start {self.start}")
        if time > _now():
            raise ValueError(f"end time {time} is in the future (now is {_now()})")
        self._end = time

    async def extend(self, cycles: int, clock: Any = None) -> None:
        """Wait `cycles` rising clock edges, then close the segment at that time.

        Call it as the last statement of the body. It is the only place where the library awaits.
        """
        clock = clock if clock is not None else self._session.clock
        if clock is None:
            raise ValueError("extend() needs a clock: pass clock= here or to Tvla")
        if cycles < 1:
            raise ValueError("cycles must be at least 1")
        await ClockCycles(clock, cycles)
        self._end = _now()


class Tvla:
    """A TVLA session. See the module documentation."""

    def __init__(
        self,
        dut: Any,
        classes: Mapping[int, Any],
        *,
        weights: Mapping[int, float] | None = None,
        warmup: int = 1,
        schedule: str = "iid",
        group: int = 0,
        groups: Mapping[int, str] | None = None,
        labels: Mapping[int, str] | None = None,
        clock: Any = None,
        design_random: Callable | None = None,
        design_random_mode: str | None = None,
        seed: int | None = None,
        batch: str | None = None,
        out: str | os.PathLike | None = None,
        meta_name: str = "meta.json",
        waveform: str | None = None,
        config: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> None:
        if not classes:
            raise ValueError("classes must not be empty")
        for label in classes:
            if not isinstance(label, int) or not 0 <= label <= MAX_LABEL:
                raise ValueError(f"class label {label!r} must be an integer in 0..={MAX_LABEL}")
        if schedule not in ("iid", "blocks"):
            raise ValueError(f"unknown schedule {schedule!r}; use 'iid' or 'blocks'")
        if warmup < 0:
            raise ValueError("warmup must not be negative")
        self.dut = dut
        self.clock = clock
        self.group = group
        self.warmup = warmup
        self.schedule = schedule
        self._classes = dict(classes)
        self._takes_rng = {k: callable(v) and _passes_rng(v) for k, v in self._classes.items()}
        self.weights = dict(weights) if weights is not None else {k: 1 for k in self._classes}
        if set(self.weights) != set(self._classes):
            raise ValueError("weights must have exactly the same labels as classes")
        make_schedule(random.Random(0), self.weights, 0, schedule)  # validates the weights

        base = seed if seed is not None else _env_int(ENV_SEED)
        if base is None:
            base = int(getattr(cocotb, "RANDOM_SEED", 0))
        self.batch = batch if batch is not None else os.environ.get(ENV_BATCH) or "b0000"
        self.seed = base
        self._seeds = derived_seeds(base, self.batch)
        self._streams = make_streams(base, self.batch)

        requested = design_random_mode
        if requested is None:
            requested = os.environ.get(ENV_DESIGN_RANDOM) or None
        if requested is not None and requested not in ("on", "off"):
            raise ValueError(f"design random mode must be 'on' or 'off', got {requested!r}")
        if requested is not None and design_random is None:
            raise ValueError(
                f"design randomness {requested!r} was requested, but there is no design_random hook"
            )
        self._design_hook = design_random
        self._design_mode = requested

        out_dir = Path(out if out is not None else os.environ.get(ENV_OUT) or ".")
        if waveform is None:
            waveform = os.environ.get(ENV_WAVEFORM) or None
        names = dict(labels) if labels is not None else {
            k: {0: "fixed", 1: "random"}.get(k, f"class{k}") for k in self._classes
        }
        group_names = dict(groups) if groups is not None else {
            group: "default" if group == 0 else f"group{group}"
        }
        design: dict[str, Any] = {}
        toplevel = getattr(dut, "_name", None)
        if isinstance(toplevel, str):
            design["toplevel"] = toplevel
        if config is not None:
            design["config"] = config
        self.meta_path = out_dir / meta_name
        self._writer = MetaWriter(
            self.meta_path,
            time=(1, _time_precision()),
            labels=names,
            groups=group_names,
            seeds={"base": base, **self._seeds},
            batch_id=self.batch,
            design=design or None,
            waveform=waveform,
            design_random={
                "requested": requested or "none", "applied": "none", "how": "none"
            },
            extensions=extensions,
        )
        self._managed = False
        self._started = False
        self._finished = False

    # -- streams --------------------------------------------------------------------------

    @property
    def rng(self) -> random.Random:
        """The stimulus stream."""
        return self._streams["stimulus"]

    @property
    def idle_rng(self) -> random.Random:
        """The stream for idle policies (random idle cycles and similar)."""
        return self._streams["idle"]

    @property
    def design_rng(self) -> random.Random:
        """The stream for design randomness. Hooks get this one."""
        return self._streams["design"]

    # -- finishing ------------------------------------------------------------------------

    def finish(self, passed: bool) -> None:
        """Write the metadata: committed if `passed`, else diagnostic. Synchronous, runs once.

        It never awaits, so it is safe in a `finally` block, also when the test times out.
        """
        if self._finished:
            return
        self._finished = True
        if passed:
            try:
                self._writer.commit()
                return
            except BaseException:
                self._writer.write_diagnostic()
                raise
        self._writer.write_diagnostic()

    def __enter__(self) -> "Tvla":
        self._managed = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(exc_type is None)
        return False

    # -- the schedule ---------------------------------------------------------------------

    async def segments(self, num_tests: int | None = None) -> AsyncIterator[Segment]:
        """Yield the warm-up segments, then `num_tests` scheduled segments.

        Without `with tvla:`, the metadata is written when this generator ends: committed if
        the loop ran to the end, diagnostic otherwise. That needs the generator to be closed when
        the loop body raises. This works in cocotb 2.1 (tested), but it depends on garbage
        collection. Use `with tvla:` for a result that does not. With `with tvla:`, a failure
        after the loop also gives a diagnostic.
        """
        env_n = _env_int(ENV_TESTS)
        if env_n is not None:
            num_tests = env_n
        if num_tests is None or num_tests < 0:
            raise ValueError("number of tests missing: pass num_tests or set " + ENV_TESTS)
        if self._started:
            raise RuntimeError("segments() can be used once per session")
        self._started = True
        completed = False
        try:
            await self._apply_design_random()
            if not all(self._takes_rng[k] for k, v in self._classes.items() if callable(v)):
                # Legacy generators use the global `random`. Seed it once per batch.
                random.seed(self.rng.getrandbits(64))
            labels = make_schedule(self._streams["schedule"], self.weights, num_tests, self.schedule)
            self._writer.skip_ids(self.warmup)
            warm_label = min(self._classes)
            for i in range(self.warmup):
                seg = Segment(self, i, i, warm_label, True, self._streams["warmup"])
                seg.start = _now()
                yield seg
                self._close(seg, record=False)
            for i, label in enumerate(labels):
                seg = Segment(self, i, self.warmup + i, label, False, self.rng)
                seg.start = _now()
                yield seg
                self._close(seg, record=True)
            completed = True
        finally:
            if not self._managed:
                self.finish(completed)

    def _close(self, seg: Segment, record: bool) -> None:
        end = seg._end if seg._end is not None else _now()
        if end <= seg.start:
            raise ValueError(
                f"segment {seg.id} has zero length (start {seg.start}, end {end}): "
                "no simulation time passed in the loop body"
            )
        if record:
            self._writer.segment(seg.start, end, seg.label, seg.group)

    async def _apply_design_random(self) -> None:
        if self._design_mode is None:
            return
        hook = self._design_hook
        result = hook(self.design_rng, self._design_mode)
        if inspect.isawaitable(result):
            result = await result
        how = f"hook: {getattr(hook, '__name__', type(hook).__name__)}"
        if isinstance(result, Mapping):
            applied, how = result["applied"], result.get("how", how)
        else:
            applied = result
        if not isinstance(applied, str):
            raise ValueError("the design_random hook must return the applied mode as a string")
        self._writer.set_design_random(
            {"requested": self._design_mode, "applied": applied, "how": str(how)}
        )
