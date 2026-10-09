import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
# The simulator imports the testbench modules from here (the runner copies sys.path).
sys.path.insert(0, str(HERE / "sim"))


def _sim_missing():
    try:
        import cocotb
    except ImportError:
        return "needs cocotb"
    if tuple(int(p) for p in cocotb.__version__.split(".")[:2]) < (2, 1):
        return "needs cocotb 2.1 or later"
    if shutil.which("verilator") is None:
        return "needs Verilator"
    return None


@pytest.fixture
def sim_available():
    reason = _sim_missing()
    if reason:
        pytest.skip(reason)


# Variables that would override what the runner sets (E2: os.environ wins over arguments).
LEAKING = ("COCOTB_", "PYGPI_", "GPI_", "TOPLEVEL", "MODULE", "TESTCASE", "WAVES", "SCASIM_TVLA_")


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory):
    from cocotb_tools.runner import get_runner

    if _sim_missing():
        pytest.skip(_sim_missing())
    d = tmp_path_factory.mktemp("build")
    get_runner("verilator").build(
        sources=[HERE / "sim" / "tiny.sv"], hdl_toplevel="tiny", build_dir=d,
        build_args=["--trace-fst", "-Wno-fatal"], waves=True, always=True,
    )
    return d


@pytest.fixture
def run_tb(build_dir, tmp_path, monkeypatch):
    from cocotb_tools.runner import get_results, get_runner

    for k in [k for k in os.environ if k.startswith(LEAKING)]:
        monkeypatch.delenv(k)

    def run(testcase, seed=1, batch="b0001", tests=None, module="tb_session"):
        out = tmp_path / f"{testcase}-{batch}-s{seed}-{len(list(tmp_path.iterdir()))}"
        out.mkdir()
        monkeypatch.setenv("SCASIM_TVLA_OUT", str(out))
        monkeypatch.setenv("SCASIM_TVLA_SEED", str(seed))
        monkeypatch.setenv("SCASIM_TVLA_BATCH", batch)
        monkeypatch.setenv("SCASIM_TVLA_WAVEFORM", "tvla.fst")
        if tests is not None:
            monkeypatch.setenv("SCASIM_TVLA_TESTS", str(tests))
        results_xml = out / "results.xml"
        try:  # the runner exits with status 1 when a test fails
            get_runner("verilator").test(
                hdl_toplevel="tiny", hdl_toplevel_lang="verilog", test_module=module,
                testcase=testcase, build_dir=build_dir, test_dir=out, waves=True,
                test_args=["--trace-file", "tvla.fst"], results_xml=str(results_xml),
            )
        except SystemExit:
            pass
        n_tests, n_fail = get_results(results_xml)
        assert n_tests == 1
        return out, n_fail

    return run




@pytest.fixture(scope="session")
def tvla_bin():
    """The tvla binary: SCASIM_TVLA_BIN, else PATH, else one `cargo build --release`."""
    import subprocess

    reason = _sim_missing()  # skip before any cargo build: CI has no simulator
    if reason:
        pytest.skip(reason)
    env = os.environ.get("SCASIM_TVLA_BIN")
    if env and Path(env).is_file():
        return Path(env)
    found = shutil.which("tvla")
    if found:
        return Path(found)
    root = next((p for p in HERE.parents if (p / "Cargo.toml").exists()
                 and (p / "src" / "bin" / "tvla").exists()), None)
    if root is None or shutil.which("cargo") is None:
        pytest.skip("needs a tvla binary: set SCASIM_TVLA_BIN or install cargo")
    done = subprocess.run(["cargo", "build", "--release", "--bin", "tvla"], cwd=root,
                          capture_output=True, text=True)
    exe = root / "target" / "release" / "tvla"
    if done.returncode != 0 or not exe.exists():
        pytest.skip(f"cannot build tvla: {done.stderr[-300:]}")
    return exe
