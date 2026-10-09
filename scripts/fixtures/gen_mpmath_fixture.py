"""Generate high-precision reference values of -log10(p) for the chi-squared survival function.

Run it from the repository root, in a virtual environment that has the pinned version:

    python3 -m venv VENV && VENV/bin/pip install mpmath==1.4.1
    VENV/bin/python -I scripts/fixtures/gen_mpmath_fixture.py [OUTPUT_FILE]

Pinned version: mpmath 1.4.1 (the output file records it).
The default output file is `tests/fixtures/stats/pvalue_mpmath.json`.
Each point is computed at two precisions. The script stops if they disagree.
"""

import json
import math
import sys
from pathlib import Path

import mpmath as mp


def neg_log10_p(dof: int, x: float, dps: int) -> mp.mpf:
    with mp.workdps(dps):
        a = mp.mpf(dof) / 2
        q = mp.gammainc(a, mp.mpf(x) / 2, mp.inf, regularized=True)
        return -mp.log10(q)


def main() -> None:
    dofs = [1, 2, 3, 4, 5, 7, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    points = set()
    for dof in dofs:
        s = math.sqrt(2 * dof)
        # Around the mean, in standard deviations.
        for k in [-2, -1, 0, 0.5, 1, 2, 3, 5, 8, 12, 20, 50]:
            x = dof + k * s
            if x > 0:
                points.add((dof, x))
        # The switch between the series and the continued fraction is at x = dof + 2.
        for d in [-1.0, -1e-6, 0.0, 1e-6, 1.0]:
            points.add((dof, dof + 2 + d))
        # Small statistics and very large statistics.
        for f in [0.01, 0.1, 0.5, 0.9]:
            points.add((dof, f * dof))
        for x in [1e3, 1e4, 1e5, 3e5]:
            if x > dof:
                points.add((dof, x))
    out = []
    for dof, x in sorted(points):
        lo = neg_log10_p(dof, x, 40)
        hi = neg_log10_p(dof, x, 80)
        err = abs(lo - hi) / max(abs(hi), mp.mpf("1e-30"))
        if err > mp.mpf("1e-25") and abs(hi) > 1e-20:
            sys.exit(f"mpmath precision check failed at dof={dof} x={x}: {lo} vs {hi}")
        out.append({"dof": dof, "x": x, "neg_log10_p": mp.nstr(hi, 30)})
    default = Path(__file__).resolve().parent.parent.parent / "tests/fixtures/stats/pvalue_mpmath.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    path.write_text(json.dumps({"mpmath": mp.__version__, "points": out}, indent=0))
    print(f"wrote {len(out)} points to {path}; max -log10 p = "
          f"{max(float(p['neg_log10_p']) for p in out):.6g}")


if __name__ == "__main__":
    main()
