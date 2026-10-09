import importlib

import scasim_tvla


def test_version():
    assert scasim_tvla.__version__ == "0.1.0"


def test_cli_stub_returns_nonzero():
    cli = importlib.import_module("scasim_tvla.cli")
    assert cli.main() != 0
