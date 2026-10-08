#!/usr/bin/env python3
"""Compare the traces and labels of two traces.npz files written by tvla.

Usage: python3 -I scripts/compare_npz.py REFERENCE.npz NEW.npz
Exit status 0 if they are identical, 1 if not.
"""
import sys

import numpy as np


def load(path):
    data = np.load(path)
    names = sorted((n for n in data.files if n.startswith("trace_")), key=lambda n: int(n[6:]))
    return np.stack([data[n] for n in names]), data["labels"]


def main():
    ref_traces, ref_labels = load(sys.argv[1])
    new_traces, new_labels = load(sys.argv[2])
    same_labels = np.array_equal(ref_labels, new_labels)
    same_shape = ref_traces.shape == new_traces.shape
    same_traces = same_shape and np.array_equal(ref_traces, new_traces)
    print(f"shapes: {ref_traces.shape} vs {new_traces.shape}")
    print(f"labels identical: {same_labels}")
    if same_shape:
        diff = np.abs(ref_traces - new_traces)
        print(f"max abs diff: {diff.max()}, differing values: {int((diff > 0).sum())}")
    print(f"traces identical: {same_traces}")
    sys.exit(0 if same_labels and same_traces else 1)


if __name__ == "__main__":
    main()
