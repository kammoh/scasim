import importlib

import pytest

import scasim_tvla


def test_version():
    assert scasim_tvla.__version__ == "0.1.0"


def test_cli_without_a_command_fails():
    cli = importlib.import_module("scasim_tvla.cli")
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0
