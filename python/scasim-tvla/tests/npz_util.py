"""Read the arrays of an .npz file without numpy (the tests add no dependency)."""

import ast
import struct
import zipfile
from array import array


def read_member(path, name):
    """The raw bytes of the array `name` (with or without the .npy suffix)."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        return z.read(name if name in names else name + ".npy")


_TYPES = {"<f8": "d", "<u4": "I", "<u2": "H", "<u8": "Q", "|u1": "B"}


def read_array(path, name, descr):
    """(shape, flat list) of a little-endian array with the numpy type `descr` (for example `<u4`)."""
    raw = read_member(path, name)
    assert raw[:6] == b"\x93NUMPY"
    if raw[6] == 1:
        (n,), start = struct.unpack("<H", raw[8:10]), 10
    else:
        (n,), start = struct.unpack("<I", raw[8:12]), 12
    header = ast.literal_eval(raw[start : start + n].decode("latin1"))
    assert header["descr"] == descr and not header["fortran_order"], header
    values = array(_TYPES[descr])
    assert values.itemsize == int(descr[2:])
    values.frombytes(raw[start + n :])
    return header["shape"], list(values)


def read_f64(path, name):
    """(shape, flat list of floats) of a little-endian float64 array."""
    return read_array(path, name, "<f8")


def members(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in sorted(z.namelist())}


def flagged(path, threshold=4.5, name="t_values"):
    """{order: sorted sample indices with |t| above the threshold}; undefined values are skipped."""
    shape, values = read_f64(path, name)
    rows, cols = shape
    return {
        d + 1: [j for j in range(cols) if abs(values[d * cols + j]) > threshold]
        for d in range(rows)
    }
