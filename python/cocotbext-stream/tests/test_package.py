"""The package imports as a namespace package under `cocotbext`."""

import importlib


def test_import_stream():
    stream = importlib.import_module("cocotbext.stream")
    assert stream.__version__ == "0.1.0"


def test_cocotbext_is_a_namespace_package():
    cocotbext = importlib.import_module("cocotbext")
    assert getattr(cocotbext, "__file__", None) is None
