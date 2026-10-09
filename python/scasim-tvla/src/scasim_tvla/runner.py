"""Build, simulate, analyze, and merge the batches of a TVLA run.

This module needs only the standard library. The simulator work (`cocotb_tools.runner`) runs in
a child process, `python -m scasim_tvla._sim`, so a run imports cocotb only when it simulates.

Layout of an output directory `OUT`:

    OUT/build/              the Verilator build and the build key
    OUT/trace.vlt           the generated trace scope rules (if any)
    OUT/bNNNN/              one batch: meta.json, tvla.fst, statistics.bin, sim.log, tvla.log
    OUT/manifest.json       the state of each batch
    OUT/meta.list           the committed batches that still have a waveform, in batch-id order
    OUT/report/             the merged result of `tvla --merge-stats`
"""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import _args
from .meta import read_meta

MANIFEST = "manifest.json"
CACHE_NAME = "statistics.bin"
TRACE_FILE = "tvla.fst"
PLACEHOLDER_WAVEFORM_BYTES = 500_000_000
# An unmeasured assumption for the default job count: RAM per simulation job.
RAM_PER_JOB = 2_000_000_000

_STRIP_PREFIXES = ("COCOTB_", "SCASIM_TVLA_")
_STRIP_NAMES = frozenset({"TOPLEVEL", "TOPLEVEL_LANG", "MODULE", "TESTCASE", "WAVES", "GUI"})


class RunnerError(Exception):
    """A run cannot continue. The message says why."""


# -- small helpers --------------------------------------------------------------------------


def clean_env(env: dict[str, str]) -> dict[str, str]:
    """Remove the variables that would override what the runner sets (E12: the shell wins)."""
    return {
        k: v for k, v in env.items()
        if not k.startswith(_STRIP_PREFIXES) and k not in _STRIP_NAMES
    }


def batch_ids(count: int) -> list[str]:
    """Zero-padded ids, so that the text order is the numeric order."""
    width = max(4, len(str(max(count - 1, 0))))
    return [f"b{i:0{width}d}" for i in range(count)]


def batch_seed(base: int, batch: str) -> int:
    """The simulation seed of one batch. It depends only on the base seed and the batch id."""
    digest = hashlib.sha256(f"scasim-tvla:{base}:{batch}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def waveform_estimate(largest_seen: int) -> int:
    """Bytes to plan for one waveform: the largest one seen, or the 0.5 GB placeholder."""
    return largest_seen if largest_seen > 0 else PLACEHOLDER_WAVEFORM_BYTES


def plan_jobs(requested: int | None, cores: int, free_disk: int, total_ram: int,
              largest_seen: int = 0) -> int:
    """Number of parallel simulations. The user's value wins. Else cores, RAM, and disk limit it.

    Disk is budgeted for `2 * jobs` waveforms (running plus waiting). The size of one is the
    placeholder until a real waveform has been seen (see `waveform_estimate`).
    """
    if requested is not None:
        if requested < 1:
            raise ValueError("--jobs must be at least 1")
        return requested
    by_disk = free_disk // (2 * waveform_estimate(largest_seen))
    by_ram = total_ram // RAM_PER_JOB if total_ram > 0 else cores
    return max(1, min(cores, by_ram, by_disk))


def find_tvla(explicit: str | None) -> Path:
    """The `tvla` binary: `--tvla PATH`, then `SCASIM_TVLA_BIN`, then PATH."""
    if explicit:
        path = Path(explicit)
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise RunnerError(f"--tvla {explicit}: not an executable file")
        return path
    env = os.environ.get("SCASIM_TVLA_BIN")
    if env:
        path = Path(env)
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise RunnerError(f"SCASIM_TVLA_BIN={env}: not an executable file")
        return path
    found = shutil.which("tvla")
    if found:
        return Path(found)
    raise RunnerError(
        "cannot find the tvla binary: use --tvla PATH, set SCASIM_TVLA_BIN, or put tvla on PATH "
        "(build it with `cargo build --release --bin tvla` in the scasim repository)"
    )


def check_signals(stdout: str, stderr: str, clock: str | None) -> int:
    """Check the output of `tvla --list-signals`. Return the number of selected signals.

    `tvla` exits with 0 even when nothing is selected or a rule matches nothing (E4), so the
    runner reads the lines. Any warning is an error. A wrong clock name is an error too.
    """
    selected = 0
    listed: set[str] = set()
    for line in stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        if parts[1] == "yes":
            selected += 1
        for name in parts[2].split(", "):
            listed.add(name.removesuffix(" (alias)"))
    warnings = [ln for ln in stderr.splitlines() if ln.lower().startswith("warning")]
    if warnings:
        raise RunnerError("the signal selection has warnings:\n  " + "\n  ".join(warnings))
    if selected == 0:
        raise RunnerError("no signal is selected: check the --include and --exclude rules")
    if clock is not None and clock not in listed:
        raise RunnerError(
            f"the clock {clock!r} is not a signal of the waveform (see tvla --list-signals)"
        )
    return selected


# -- sources, trace scope rules, build key ----------------------------------------------------


def parse_sources(items: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Read source files and file lists. Return (source files, extra Verilator arguments).

    A `.f` or `.list` file has one path per line, relative to the list. A line that starts with
    `-` or `+` holds Verilator arguments. `#` starts a comment.
    """
    files: list[Path] = []
    args: list[str] = []
    for item in items:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(f"source {item} does not exist")
        if path.suffix not in (".f", ".list"):
            files.append(path)
            continue
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line[0] in "-+":
                args += shlex.split(line)
                continue
            src = path.parent / line
            if not src.is_file():
                raise FileNotFoundError(f"{path}: source {line} does not exist")
            files.append(src)
    return files, args


def _check_scope(name: str) -> str:
    if not name or not name.strip() or any(c in name for c in '"\n\r'):
        raise ValueError(f"bad trace scope name {name!r}")
    if name == "TOP" or name.startswith("TOP."):
        raise ValueError(
            f"trace scope {name!r}: Verilator scope names have no TOP prefix (use {name[4:]!r})"
        )
    return name


def vlt_text(toplevel: str, scopes: Sequence[str], off_rules: Sequence[str]) -> str | None:
    """The Verilator control file for the trace scope rules (E4 syntax), or None without rules."""
    if not scopes and not off_rules:
        return None
    lines = ["`verilator_config"]
    if scopes:
        lines.append(f'tracing_off -scope "{_check_scope(toplevel)}"')
        lines += [f'tracing_on -scope "{_check_scope(s)}"' for s in scopes]
    lines += [f'tracing_off -scope "{_check_scope(r)}"' for r in off_rules]
    return "\n".join(lines) + "\n"


def verilator_args(user_args: Sequence[str], trace_depth: int | None) -> list[str]:
    """`-O3 --trace-fst`, then the user's arguments. No --trace-threads, no --x-initial."""
    args = ["-O3", "--trace-fst", *user_args]
    if trace_depth is not None:
        args += ["--trace-depth", str(trace_depth)]
    return args


def build_key(sources: Sequence[Path], toplevel: str, build_args: Sequence[str],
              vlt: str | None, verilator: str, cocotb: str) -> str:
    """A hash of everything that changes the build."""
    h = hashlib.sha256()

    def put(*parts: str | bytes) -> None:
        for part in parts:
            data = part.encode() if isinstance(part, str) else part
            h.update(len(data).to_bytes(8, "big"))
            h.update(data)

    put("scasim-tvla-build-1", toplevel, verilator, cocotb, json.dumps(list(build_args)))
    put(vlt or "")
    for src in sources:
        put(str(src.resolve()), src.read_bytes())
    return h.hexdigest()


def tool_versions() -> tuple[str, str]:
    """(Verilator version text, cocotb version). Imports nothing from cocotb."""
    from importlib import metadata

    try:
        cocotb = metadata.version("cocotb")
    except metadata.PackageNotFoundError:
        raise RunnerError("cocotb 2.1 or later is required to simulate (pip install cocotb)") from None
    major_minor = tuple(int(p) for p in re.findall(r"\d+", cocotb)[:2])
    if major_minor < (2, 1):
        raise RunnerError(f"cocotb {cocotb} is too old: scasim-tvla needs cocotb 2.1 or later")
    exe = shutil.which("verilator")
    if exe is None:
        raise RunnerError("Verilator is not on PATH")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip(), cocotb


# -- manifest ---------------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


class Manifest:
    """The state of each batch, saved after every change. Thread-safe."""

    STATES = ("simulated", "cached", "failed")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {"manifest": 1, "batches": {}}
        if path.exists():
            self.data = json.loads(path.read_text())

    def get(self, batch: str) -> dict[str, Any]:
        with self._lock:
            return dict(self.data["batches"].get(batch, {}))

    def set(self, batch: str, **fields: Any) -> None:
        if fields.get("state") not in (None, *self.STATES):
            raise ValueError(f"unknown batch state {fields['state']!r}")
        with self._lock:
            rec = self.data["batches"].setdefault(batch, {})
            rec.update(fields)
            _write_json_atomic(self.path, self.data)

    def set_root(self, **fields: Any) -> None:
        with self._lock:
            self.data.update(fields)
            _write_json_atomic(self.path, self.data)

    def reset_batches(self) -> None:
        with self._lock:
            self.data["batches"] = {}
            _write_json_atomic(self.path, self.data)

    def batches(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in sorted(self.data["batches"].items())}


# -- configuration and result -----------------------------------------------------------------


@dataclass
class SimSpec:
    """What the simulator side needs. The tests may leave most fields empty."""

    sources: list[str] = field(default_factory=list)
    toplevel: str = ""
    test_module: str = ""
    testcase: str | None = None
    build_args: list[str] = field(default_factory=list)  # user's Verilator arguments
    trace_scopes: list[str] = field(default_factory=list)
    trace_off_rules: list[str] = field(default_factory=list)
    trace_depth: int | None = None
    pythonpath: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    out: Path
    batches: int
    tests_per_batch: int
    seed: int | None = None
    jobs: int | None = None
    keep: str = "none"  # none | waveform
    analyze: bool = True
    tvla: str | None = None
    tvla_args: list[str] = field(default_factory=list)
    curve: str | None = None
    profile: bool = False
    design_random: str | None = None
    sim: SimSpec = field(default_factory=SimSpec)


@dataclass
class SimResult:
    """What a simulation attempt reports. The runner decides success from the metadata."""

    exit_code: int | None = 0
    tests: int = 0
    fails: int = 0
    error: str | None = None
    cpu: dict[str, float] | None = None  # only with --profile


@dataclass
class RunReport:
    ok: list[str]
    failed: dict[str, str]
    report_dir: Path | None
    merge_command: list[str] | None = None

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def meta_file(batch_dir: Path) -> Path:
    """`meta.json`, or `meta.json.gz` if only that exists (Makefile flows)."""
    gz = batch_dir / "meta.json.gz"
    plain = batch_dir / "meta.json"
    return gz if gz.exists() and not plain.exists() else plain


def _log(msg: str) -> None:
    print(f"scasim-tvla: {msg}", file=sys.stderr, flush=True)


def _tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# -- the pipeline -----------------------------------------------------------------------------


class Pipeline:
    """Run all batches of a configuration. `simulate` replaces the child process (for tests)."""

    def __init__(self, cfg: RunConfig,
                 simulate: Callable[["Pipeline", str, dict[str, Any]], SimResult] | None = None):
        self.cfg = cfg
        self.out = Path(cfg.out)
        self.manifest = Manifest(self.out / MANIFEST)
        self._simulate = simulate
        self.parts = _args.partition(cfg.tvla_args, need_clock=cfg.analyze)
        if cfg.curve is not None:
            _args.check_curve(cfg.curve)
        if cfg.keep not in ("none", "waveform"):
            raise RunnerError(f"--keep {cfg.keep!r}: only none and waveform are supported")
        self.tvla_bin: Path | None = None
        self.build_dir: Path | None = None
        self.build_hash = ""
        self.largest_waveform = 0
        self._lock = threading.Lock()
        self.jobs = 1

    # -- setup --------------------------------------------------------------------------

    def _config_hash(self) -> str:
        c = self.cfg
        payload = {
            "seed": self.seed, "tests": c.tests_per_batch, "design_random": c.design_random,
            "build": self.build_hash, "module": c.sim.test_module, "testcase": c.sim.testcase,
            "preprocess": self.parts.preprocess, "toplevel": c.sim.toplevel,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def prepare(self) -> None:
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        try:
            self.tvla_bin = find_tvla(cfg.tvla)
        except RunnerError:
            if cfg.analyze:
                raise
            _log("warning: no tvla binary found, so the signal check is skipped")
        if cfg.keep == "none" and not cfg.analyze:
            cfg.keep = "waveform"  # nothing else would hold the data
        if self._simulate is None:
            self.build_dir = self._build()
        total_ram = 0
        try:
            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            pass
        seen = max((r.get("waveform_bytes", 0) for r in self.manifest.batches().values()), default=0)
        self.largest_waveform = seen
        self.jobs = plan_jobs(cfg.jobs, os.cpu_count() or 1,
                              shutil.disk_usage(self.out).free, total_ram, largest_seen=seen)
        base = cfg.seed
        if base is None:
            base = self.manifest.data.get("seed")
        if base is None:
            base = int.from_bytes(os.urandom(4), "big")
        self.seed = int(base)
        old = self.manifest.data.get("config")
        new = self._config_hash()
        if old is not None and old != new:
            _log("the configuration changed since the last run: all batches run again")
            self.manifest.reset_batches()
        self.manifest.set_root(seed=self.seed, config=new, tvla_args=list(cfg.tvla_args),
                               curve=cfg.curve, keep=cfg.keep, tests_per_batch=cfg.tests_per_batch)

    def _build(self) -> Path:
        sim = self.cfg.sim
        files, list_args = parse_sources(sim.sources)
        vlt = vlt_text(sim.toplevel, sim.trace_scopes, sim.trace_off_rules)
        if sim.trace_depth is not None:
            _log("warning: --trace-depth does not select the DUT scope reliably (E4); "
                 "use --trace-scope")
        sources = [str(f.resolve()) for f in files]
        if vlt is not None:
            vlt_path = self.out / "trace.vlt"
            vlt_path.write_text(vlt)
            sources.append(str(vlt_path.resolve()))
        build_args = verilator_args([*list_args, *sim.build_args], sim.trace_depth)
        verilator, cocotb = tool_versions()
        key = build_key(files, sim.toplevel, build_args, vlt, verilator, cocotb)
        self.build_hash = key
        build_dir = (self.out / "build").resolve()
        key_file = build_dir / "scasim-tvla-build.json"
        try:
            same = json.loads(key_file.read_text()).get("key") == key
        except (OSError, ValueError):
            same = False
        if same and (build_dir / sim.toplevel).exists():
            _log("reusing the build")
            return build_dir
        _log("building the design with Verilator")
        spec = {"mode": "build", "sources": sources, "toplevel": sim.toplevel,
                "build_args": build_args, "build_dir": str(build_dir),
                "pythonpath": sim.pythonpath}
        spec_path = self.out / "build-spec.json"
        _write_json_atomic(spec_path, spec)
        log = self.out / "build.log"
        code = self._run_worker("build", spec_path, log)
        if code != 0 or not (build_dir / sim.toplevel).exists():
            raise RunnerError(f"the build failed (exit {code}); see {log}:\n{_tail(log)}")
        _write_json_atomic(key_file, {"key": key, "verilator": verilator, "cocotb": cocotb})
        return build_dir

    def _run_worker(self, mode: str, spec_path: Path, log: Path) -> int:
        env = clean_env(dict(os.environ))
        with open(log, "w") as fh:
            proc = subprocess.run(
                [sys.executable, "-m", "scasim_tvla._sim", mode, str(spec_path)],
                env=env, stdout=fh, stderr=subprocess.STDOUT, check=False,
            )
        return proc.returncode

    # -- one simulation -----------------------------------------------------------------

    def batch_dir(self, batch: str) -> Path:
        return self.out / batch

    def sim_env(self, batch: str, seed: int, tests: int) -> dict[str, str]:
        """The variables that the testbench side reads (`scasim_tvla.session`)."""
        env = {
            "SCASIM_TVLA_OUT": str(self.batch_dir(batch).resolve()),
            "SCASIM_TVLA_SEED": str(seed),
            "SCASIM_TVLA_BATCH": batch,
            "SCASIM_TVLA_TESTS": str(tests),
            "SCASIM_TVLA_WAVEFORM": TRACE_FILE,
        }
        if self.cfg.design_random is not None:
            env["SCASIM_TVLA_DESIGN_RANDOM"] = self.cfg.design_random
        if self.cfg.profile:
            env["COCOTB_ENABLE_PROFILING"] = "1"
        return env

    def simulate_batch(self, batch: str, tests: int) -> SimResult:
        d = self.batch_dir(batch)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        seed = batch_seed(self.seed, batch)
        info = {"batch": batch, "seed": seed, "tests": tests, "dir": d,
                "env": self.sim_env(batch, seed, tests)}
        if self._simulate is not None:
            return self._simulate(self, batch, info)
        sim = self.cfg.sim
        spec = {
            "mode": "test", "toplevel": sim.toplevel, "test_module": sim.test_module,
            "testcase": sim.testcase, "build_dir": str(self.build_dir), "test_dir": str(d.resolve()),
            "seed": seed, "env": info["env"], "trace_file": TRACE_FILE,
            "profile": self.cfg.profile, "pythonpath": sim.pythonpath,
        }
        spec_path = d / "sim-spec.json"
        _write_json_atomic(spec_path, spec)
        code = self._run_worker("test", spec_path, d / "sim.log")
        try:
            data = json.loads((d / "sim-result.json").read_text())
        except (OSError, ValueError):
            return SimResult(exit_code=code, error=f"no result from the simulation worker (exit {code})")
        return SimResult(exit_code=data.get("exit"), tests=data.get("tests", 0),
                         fails=data.get("fails", 0), error=data.get("error"), cpu=data.get("cpu"))

    def check_batch(self, batch: str, result: SimResult) -> str | None:
        """None if the batch is usable, else the reason. The metadata decides, not the exit code."""
        d = self.batch_dir(batch)
        meta_path = meta_file(d)
        if result.error:
            return result.error
        if result.tests != 1 or result.fails != 0:
            return f"cocotb reported {result.tests} tests and {result.fails} failures"
        if not meta_path.exists():
            return "the testbench wrote no meta.json"
        try:
            meta = read_meta(meta_path)
        except Exception as exc:  # noqa: BLE001 - any read problem makes the batch unusable
            return f"cannot read meta.json: {exc}"
        status = meta.get("batch", {}).get("status")
        if status != "committed":
            reason = meta.get("extensions", {}).get("diagnostic", {}).get("reason")
            return f"the metadata status is {status!r}" + (f" ({reason})" if reason else "")
        wave = d / TRACE_FILE
        if not wave.exists() or wave.stat().st_size == 0:
            return "the waveform is missing or empty"
        return None

    # -- tvla calls ---------------------------------------------------------------------

    def _tvla(self, args: list[str], log: Path, cwd: Path | None = None) -> tuple[int, str, str]:
        assert self.tvla_bin is not None
        proc = subprocess.run([str(self.tvla_bin), *args], capture_output=True, text=True,
                              cwd=cwd, check=False)
        log.write_text(f"$ {shlex.join([str(self.tvla_bin), *args])}\n"
                       f"--- stdout\n{proc.stdout}--- stderr\n{proc.stderr}")
        return proc.returncode, proc.stdout, proc.stderr

    def check_selection(self, meta_path: Path) -> int:
        """`tvla --list-signals` on one batch, with the user's rules."""
        code, out, err = self._tvla(
            ["--meta-json", str(meta_path), "--list-signals", *self.parts.select],
            meta_path.parent / "list-signals.log",
        )
        if code != 0:
            raise RunnerError(f"tvla --list-signals failed (exit {code}):\n{err.strip()}")
        return check_signals(out, err, self.parts.clock)

    def probe(self) -> None:
        """One short run and the signal check, before the real batches start."""
        if self.tvla_bin is None:
            return
        _log("probe run and signal check")
        result = self.simulate_batch("probe", min(self.cfg.tests_per_batch, 2))
        reason = self.check_batch("probe", result)
        if reason is not None:
            raise RunnerError(f"the probe run failed: {reason} (see {self.batch_dir('probe')})")
        count = self.check_selection(meta_file(self.batch_dir("probe")))
        _log(f"{count} signals are selected")
        shutil.rmtree(self.batch_dir("probe"))

    def analyze_batch(self, batch: str) -> None:
        """`tvla --stats-out` on a committed batch. Raises RunnerError on failure."""
        d = self.batch_dir(batch)
        cache = d / CACHE_NAME
        cache.unlink(missing_ok=True)
        scratch = d / "tvla-out"
        args = ["--meta-json", str(meta_file(d)), "--stats-out", str(cache),
                *self.parts.batch, "-d", "1", "--plot=false", "--chi2=false",
                "--ttest-output-dir", str(scratch)]
        if not self.parts.group_choice:
            args.append("--pool-groups")
        code, _out, err = self._tvla(args, d / "tvla.log")
        shutil.rmtree(scratch, ignore_errors=True)
        if code != 0 or not cache.exists() or cache.stat().st_size == 0:
            cache.unlink(missing_ok=True)
            raise RunnerError(f"tvla --stats-out failed (exit {code}): {err.strip()[-400:]}")

    # -- the batch loop -----------------------------------------------------------------

    def _action(self, batch: str) -> str:
        rec = self.manifest.get(batch)
        d = self.batch_dir(batch)
        state = rec.get("state")
        if state == "cached" and (d / CACHE_NAME).exists():
            return "skip"
        if state == "simulated" and self.check_batch(batch, SimResult(tests=1)) is None:
            return "analyze" if self.cfg.analyze else "skip"
        return "simulate"

    def _free_disk_ok(self) -> bool:
        if self.largest_waveform <= 0:
            return True
        return shutil.disk_usage(self.out).free >= self.largest_waveform

    def _fail(self, batch: str, reason: str) -> None:
        _log(f"{batch} failed: {reason}")
        d = self.batch_dir(batch)
        d.mkdir(parents=True, exist_ok=True)
        (d / "diagnostic.txt").write_text(reason + "\n")
        self.manifest.set(batch, state="failed", error=reason)

    def _simulate_stage(self, batch: str) -> bool:
        """Simulate one batch. True if it is committed and ready for analysis."""
        if not self._free_disk_ok():
            self._fail(batch, "not enough free disk for another waveform")
            return False
        self.manifest.set(batch, seed=batch_seed(self.seed, batch), error=None)
        try:
            result = self.simulate_batch(batch, self.cfg.tests_per_batch)
        except Exception as exc:  # noqa: BLE001
            self._fail(batch, f"the simulation could not run: {exc}")
            return False
        reason = self.check_batch(batch, result)
        if reason is not None:
            self._fail(batch, reason)
            return False
        size = (self.batch_dir(batch) / TRACE_FILE).stat().st_size
        with self._lock:
            self.largest_waveform = max(self.largest_waveform, size)
        fields: dict[str, Any] = {"state": "simulated", "waveform_bytes": size}
        if result.cpu is not None:
            fields["sim_cpu"] = result.cpu
        self.manifest.set(batch, **fields)
        return True

    def _analyze_stage(self, batch: str) -> None:
        try:
            self.analyze_batch(batch)
        except (RunnerError, OSError) as exc:
            self._fail(batch, str(exc))
            return
        d = self.batch_dir(batch)
        self.manifest.set(batch, state="cached", cache=f"{batch}/{CACHE_NAME}",
                          cache_bytes=(d / CACHE_NAME).stat().st_size)
        self._drop_waveform(batch)

    def _drop_waveform(self, batch: str) -> None:
        if self.cfg.keep == "none":
            (self.batch_dir(batch) / TRACE_FILE).unlink(missing_ok=True)

    def run_batches(self) -> None:
        ids = batch_ids(self.cfg.batches)
        jobs = self.jobs
        waiting = threading.BoundedSemaphore(jobs)  # finished waveforms that wait for analysis
        analysis = futures.ThreadPoolExecutor(max_workers=min(jobs, 2), thread_name_prefix="tvla")
        pending: list[futures.Future] = []

        def analyze_then_release(batch: str) -> None:
            try:
                self._analyze_stage(batch)
            finally:
                waiting.release()

        def one(batch: str) -> None:
            action = self._action(batch)
            if action == "skip":
                if self.manifest.get(batch).get("state") == "cached":
                    self._drop_waveform(batch)
                return
            if action == "simulate" and not self._simulate_stage(batch):
                return
            if not self.cfg.analyze or action == "skip":
                return
            waiting.acquire()  # blocks while `jobs` waveforms wait: simulation pauses
            try:
                pending.append(analysis.submit(analyze_then_release, batch))
            except BaseException:
                waiting.release()
                raise

        with futures.ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="sim") as sims:
            for fut in [sims.submit(one, b) for b in ids]:
                fut.result()
        analysis.shutdown(wait=True)
        for fut in pending:
            fut.result()

    # -- merge and report ---------------------------------------------------------------

    def write_meta_list(self) -> None:
        lines = []
        for batch, rec in self.manifest.batches().items():
            d = self.batch_dir(batch)
            if rec.get("state") in ("simulated", "cached") and meta_file(d).exists() \
                    and (d / TRACE_FILE).exists():
                lines.append(f"{batch}/{meta_file(d).name}")
        (self.out / "meta.list").write_text("".join(f"{ln}\n" for ln in lines))

    def merge(self) -> tuple[list[str], list[str]]:
        """Merge the caches in batch-id order. Return (command, batch ids)."""
        wanted = set(batch_ids(self.cfg.batches)) if self.cfg.batches else None
        ordered = [(b, r) for b, r in self.manifest.batches().items()
                   if r.get("state") == "cached" and (wanted is None or b in wanted)]
        if not ordered:
            raise RunnerError("no batch has a statistics cache: nothing to merge")
        report = self.out / "report"
        report.mkdir(exist_ok=True)
        files = [str(self.out / r["cache"]) for _b, r in ordered]
        args = ["--merge-stats", *files, "--ttest-output-dir", str(report), *self.parts.merge]
        if self.cfg.curve is not None:
            args += ["--curve", self.cfg.curve]
        code, _out, err = self._tvla(args, report / "tvla.log")
        if code != 0:
            raise RunnerError(f"tvla --merge-stats failed (exit {code}): {err.strip()[-400:]}\n"
                              f"see {report / 'tvla.log'}")
        return [str(self.tvla_bin), *args], [b for b, _r in ordered]

    def execute(self) -> RunReport:
        self.prepare()
        if self.tvla_bin is not None:
            actions = {b: self._action(b) for b in batch_ids(self.cfg.batches)}
            if "simulate" in actions.values():
                self.probe()
            else:  # nothing to simulate: check the selection on a waveform that waits
                waiting = next((b for b, a in actions.items() if a == "analyze"), None)
                if waiting is not None:
                    self.check_selection(meta_file(self.batch_dir(waiting)))
        _log(f"running {self.cfg.batches} batches, {self.jobs} at a time, seed {self.seed}")
        self.run_batches()
        self.write_meta_list()
        batches = self.manifest.batches()
        failed = {b: r.get("error", "failed") for b, r in batches.items() if r.get("state") == "failed"}
        ok = [b for b, r in batches.items() if r.get("state") in ("simulated", "cached")]
        if not self.cfg.analyze:
            return RunReport(ok=ok, failed=failed, report_dir=None)
        command = None
        report_dir = None
        if any(r.get("state") == "cached" for r in batches.values()):
            command, _ids = self.merge()
            report_dir = self.out / "report"
            self._write_report(report_dir, command)
        elif not failed:
            raise RunnerError("no batches to merge")
        return RunReport(ok=ok, failed=failed, report_dir=report_dir, merge_command=command)

    def _write_report(self, report_dir: Path, command: list[str]) -> None:
        batches = self.manifest.batches()
        data: dict[str, Any] = {
            "seed": self.seed, "jobs": self.jobs, "keep": self.cfg.keep, "merge_command": command,
            "batches": {b: {k: r.get(k) for k in ("state", "seed", "cache_bytes", "waveform_bytes",
                                                    "error") if k in r}
                        for b, r in batches.items()},
        }
        if self.cfg.profile:
            data["sim_cpu"] = {b: r["sim_cpu"] for b, r in batches.items() if "sim_cpu" in r}
        _write_json_atomic(report_dir / "run.json", data)


# -- collect and merge for Makefile flows -------------------------------------------------------


def _batch_dirs(root: Path) -> list[Path]:
    skip = {"build", "report", "probe"}
    found = []
    def natural(p: Path) -> list:
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]

    for d in sorted((p for p in root.iterdir() if p.is_dir() and p.name not in skip), key=natural):
        if (d / "meta.json").exists() or (d / "meta.json.gz").exists():
            found.append(d)
    return found


def collect(root: Path, tvla_args: Sequence[str] = (), curve: str | None = None,
            keep: str = "waveform") -> Manifest:
    """Build `meta.list` and the manifest from batch directories that a Makefile flow made.

    Each subdirectory of `root` with a `meta.json` (or `.gz`) is a batch. A committed batch with a
    `statistics.bin` is `cached`, one with a waveform is `simulated`. Others are `failed`.
    """
    root = Path(root)
    manifest = Manifest(root / MANIFEST)
    manifest.reset_batches()
    manifest.set_root(tvla_args=list(tvla_args), curve=curve, keep=keep, seed=None, config=None)
    listed = []
    for d in _batch_dirs(root):
        mf = meta_file(d)
        try:
            meta = read_meta(mf)
        except Exception as exc:  # noqa: BLE001
            manifest.set(d.name, state="failed", error=f"cannot read {mf.name}: {exc}")
            continue
        status = meta.get("batch", {}).get("status")
        wave = meta.get("waveform")
        has_wave = bool(wave) and (d / wave).exists()
        if status != "committed":
            reason = meta.get("extensions", {}).get("diagnostic", {}).get("reason")
            manifest.set(d.name, state="failed",
                         error=f"the metadata status is {status!r}" + (f" ({reason})" if reason else ""))
        elif (d / CACHE_NAME).exists():
            manifest.set(d.name, state="cached", cache=f"{d.name}/{CACHE_NAME}",
                         cache_bytes=(d / CACHE_NAME).stat().st_size)
        elif has_wave:
            manifest.set(d.name, state="simulated", waveform_bytes=(d / wave).stat().st_size)
        else:
            manifest.set(d.name, state="failed", error="no waveform and no statistics cache")
        if status == "committed" and has_wave:
            listed.append(f"{d.name}/{mf.name}")
    (root / "meta.list").write_text("".join(f"{ln}\n" for ln in listed))
    return manifest


def merge_dir(root: Path, tvla: str | None = None, tvla_args: Sequence[str] | None = None,
              curve: str | None = None) -> RunReport:
    """Analyze the `simulated` batches of `root` that still have a waveform, then merge."""
    root = Path(root)
    manifest = Manifest(root / MANIFEST)
    if not manifest.path.exists():
        raise RunnerError(f"{root} has no {MANIFEST}: run scasim-tvla collect {root} first")
    args = list(tvla_args) if tvla_args is not None else list(manifest.data.get("tvla_args", []))
    curve = curve if curve is not None else manifest.data.get("curve")
    cfg = RunConfig(out=root, batches=0, tests_per_batch=0, tvla=tvla, tvla_args=args, curve=curve,
                    keep=manifest.data.get("keep", "waveform"))
    pipe = Pipeline(cfg, simulate=lambda *_: SimResult())
    pipe.tvla_bin = find_tvla(tvla)
    pipe.manifest = manifest
    pipe.seed = manifest.data.get("seed") or 0
    for batch, rec in manifest.batches().items():
        d = root / batch
        if rec.get("state") == "simulated" and meta_file(d).exists():
            pipe._analyze_stage(batch)
    batches = manifest.batches()
    failed = {b: r.get("error", "failed") for b, r in batches.items() if r.get("state") == "failed"}
    ok = [b for b, r in batches.items() if r.get("state") in ("simulated", "cached")]
    command, _ids = pipe.merge()
    pipe._write_report(root / "report", command)
    return RunReport(ok=ok, failed=failed, report_dir=root / "report", merge_command=command)
