"""Generate contingency tables and SciPy reference results for the Rust tests.

Run it from the repository root, in a virtual environment that has the pinned versions:

    python3 -m venv VENV && VENV/bin/pip install scipy==1.18.1 numpy==2.5.3 mpmath==1.4.1
    VENV/bin/python -I scripts/fixtures/gen_scipy_fixture.py [OUTPUT_FILE]

Pinned versions: SciPy 1.18.1, NumPy 2.5.3, mpmath 1.4.1. The output file records the SciPy and
NumPy versions. The committed fixture was made before the NumPy field existed, so it records
only SciPy.
NumPy matters because the tables come from `numpy.random.default_rng(20261008)`.
The default output file is `tests/fixtures/stats/scipy_tables.json`.

For each table, SciPy computes Pearson's statistic (`lambda_=None`) and the G statistic
(`lambda_="log-likelihood"`) with `correction=False`. SciPy raises an error for zero expected
frequencies, so all-zero rows and columns are removed before the call. The Rust code receives
the table with those rows and columns still in place, and must drop them itself.
"""

import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.stats import chi2_contingency

rng = np.random.default_rng(20261008)


def reference(table: np.ndarray) -> dict:
    keep_rows = table.sum(axis=1) > 0
    keep_cols = table.sum(axis=0) > 0
    t = table[keep_rows][:, keep_cols]
    out = {}
    if t.shape[0] < 2 or t.shape[1] < 2:
        for name in ("pearson", "g"):
            out[name] = {"stat": 0.0, "dof": 0, "p": 1.0}
        return out
    for name, lam in (("pearson", None), ("g", "log-likelihood")):
        res = chi2_contingency(t, correction=False, lambda_=lam)
        out[name] = {"stat": float(res.statistic), "dof": int(res.dof), "p": float(res.pvalue)}
    return out


def g_statistic_mpmath(table: np.ndarray) -> float:
    """G statistic with 50 digits. SciPy's float64 sum of F*log(F/E) loses digits to cancellation
    for near-null tables with large counts, so this is the reference for G."""
    mp.mp.dps = 50
    t = table[table.sum(axis=1) > 0]
    t = t[:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    rows = [[mp.mpf(int(x)) for x in r] for r in t]
    r_tot = [sum(r) for r in rows]
    c_tot = [sum(r[j] for r in rows) for j in range(t.shape[1])]
    n = sum(r_tot)
    g = mp.mpf(0)
    for i, r in enumerate(rows):
        for j, f in enumerate(r):
            if f > 0:
                g += 2 * f * mp.log(f * n / (r_tot[i] * c_tot[j]))
    return float(g)


def sample_table(n_rows: int, n_cols: int, per_row: int, effect: float) -> np.ndarray:
    base = rng.dirichlet(np.ones(n_cols) * 2.0)
    table = np.zeros((n_rows, n_cols), dtype=np.int64)
    for i in range(n_rows):
        pmf = base.copy()
        if effect > 0:
            pmf = pmf * np.exp(effect * rng.standard_normal(n_cols))
            pmf /= pmf.sum()
        table[i] = rng.multinomial(per_row, pmf)
    return table


def main() -> None:
    cases = []

    def add(name: str, table: np.ndarray) -> None:
        ref = reference(table)
        ref["g_mpmath"] = g_statistic_mpmath(table)
        cases.append({"name": name, "table": table.tolist(), **ref})

    # The worked example of Moradi et al. 2018.
    add("paper_example", np.array([[24, 59, 28, 9], [23, 57, 20, 0]]))

    # Two classes, many sizes, null and alternative.
    for n_cols in (2, 3, 5, 10, 20, 40, 64):
        for per_row in (50, 500, 5000, 100000, 3000000):
            for effect in (0.0, 0.05, 0.3):
                add(f"2x{n_cols}_n{per_row}_e{effect}", sample_table(2, n_cols, per_row, effect))

    # Unequal row totals.
    for _ in range(20):
        n_cols = int(rng.integers(2, 40))
        t = sample_table(2, n_cols, 1000, 0.1)
        t[1] = rng.multinomial(int(rng.integers(30, 20000)), rng.dirichlet(np.ones(n_cols) * 2))
        add(f"2x{n_cols}_unequal", t)

    # Several classes.
    for n_rows in (3, 4, 8, 16):
        for n_cols in (2, 5, 17):
            for per_row in (200, 20000):
                for effect in (0.0, 0.2):
                    add(f"{n_rows}x{n_cols}_n{per_row}_e{effect}", sample_table(n_rows, n_cols, per_row, effect))

    # Tables with all-zero rows and columns in the input.
    for k in range(10):
        t = sample_table(3, 6, 400, 0.2)
        t = np.insert(t, int(rng.integers(0, 4)), 0, axis=0)
        t = np.insert(t, int(rng.integers(0, 7)), 0, axis=1)
        t = np.insert(t, 0, 0, axis=1)
        add(f"zeros_{k}", t)

    # A deterministic fixed class against a spread-out random class (noise-free simulation).
    for n_cols in (8, 33):
        t = np.zeros((2, n_cols), dtype=np.int64)
        t[0, n_cols // 2] = 4000
        t[1] = rng.multinomial(4000, rng.dirichlet(np.ones(n_cols) * 3))
        add(f"fixed_vs_random_{n_cols}", t)

    # Degenerate: a single non-empty column.
    add("single_column", np.array([[0, 7, 0], [0, 9, 0]]))

    default = Path(__file__).resolve().parent.parent.parent / "tests/fixtures/stats/scipy_tables.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    path.write_text(
        json.dumps({"scipy": scipy.__version__, "numpy": np.__version__, "cases": cases})
    )
    finite_p = sum(1 for c in cases if c["pearson"]["p"] > 1e-300)
    print(f"wrote {len(cases)} tables ({finite_p} with a finite Pearson p-value) to {path}")


if __name__ == "__main__":
    main()
