"""Generate reference values of the upper-tail normal quantile for `stats::special::normal_isf`.

The values come from SciPy `norm.isf(p)`. The script also computes each value with mpmath
(50 digits) and stops if SciPy and mpmath differ by more than 1e-15 (relative).

Pinned versions: Python 3.14, NumPy 2.5.3, SciPy 1.18.1, mpmath 1.4.1.

Run from the repository root, in a virtual environment that holds these versions
(`pip install scipy==1.18.1 mpmath==1.4.1`; never install packages globally):

    venv/bin/python -I scripts/fixtures/gen_norm_isf_fixture.py

Output: tests/fixtures/stats/norm_isf.json. Each p and z is a string with 17 significant
digits, so the Rust test parses the exact double. The fixture holds about 200 log-spaced p from
1e-300 to 0.5, and the Bonferroni levels 1e-5 / (2 m) for m in M_VALUES.
"""

import json
import platform
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.stats import norm

M_VALUES = [1, 10, 371, 742, 1484, 1_000_000]
N_LOG = 200


def exact_isf(p: float) -> float:
    """Reference z with Q(z) = p, from mpmath (erfc root finding at 50 digits)."""
    with mp.workdps(50):
        target = mp.mpf(p)
        # Q(z) = erfc(z / sqrt(2)) / 2. Start from the SciPy value and refine.
        z0 = mp.mpf(float(norm.isf(p)))
        z = mp.findroot(lambda z: mp.erfc(z / mp.sqrt(2)) / 2 - target, z0, tol=1e-45)
        return float(z)


def main() -> None:
    ps = [float(10.0 ** e) for e in np.linspace(-300.0, np.log10(0.5), N_LOG)]
    ps += [1e-5 / (2 * m) for m in M_VALUES]
    ps = sorted(set(ps))
    points = []
    for p in ps:
        z = float(norm.isf(p))
        ref = exact_isf(p)
        if abs(z - ref) > 1e-15 * abs(ref):
            sys.exit(f"SciPy and mpmath differ at p = {p!r}: {z!r} versus {ref!r}")
        points.append({"p": f"{p:.16e}", "z": f"{z:.16e}"})
    out = {
        "description": "norm.isf(p): z with P(Z > z) = p. Strings have 17 significant digits.",
        "generator": "scripts/fixtures/gen_norm_isf_fixture.py (run with `python3 -I` in a venv)",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mpmath": mp.__version__,
        },
        "bonferroni_m_values": M_VALUES,
        "points": points,
    }
    path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "stats" / "norm_isf.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(points)} points to {path}")


if __name__ == "__main__":
    main()
