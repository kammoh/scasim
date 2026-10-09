"""Generate high-precision reference values of -log10(p) for the chi-squared survival function.

Run it from the repository root, in a virtual environment that has the pinned version:

    python3 -m venv VENV && VENV/bin/pip install mpmath==1.4.1
    VENV/bin/python -I scripts/fixtures/gen_mpmath_fixture.py [OUTPUT_FILE]

Pinned version: mpmath 1.4.1 (the output file records it).
The default output file is `tests/fixtures/stats/pvalue_mpmath.json`. The script also writes
`gamma_q_mpmath.json` next to it: `ln Q(a, x)` for small shapes `a` and for huge `a`.
Each point is computed at two precisions. The script stops if they disagree.

Points with dof up to 5000 use `mpmath.gammainc`. For larger dof, `gammainc` does not converge,
so the script sums the series `P(a, x) = x^a e^-x / Gamma(a + 1) * 1F1(1; a + 1; x)` with
`mpmath.hyp1f1` and takes `Q = 1 - P`. The working precision grows until two precisions agree,
because `1 - P` loses digits when `Q` is tiny.
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


LARGE_DOFS = [100_000, 1_000_000, 10_000_000, 1_000_000_000, 4_294_967_295]


def ln_q_large(a: float, x: float) -> mp.mpf:
    """ln Q(a, x) for any a > 0, to 30 digits. Two precisions must agree."""
    prev = None
    dps = 60
    while dps <= 4000:
        with mp.workdps(dps):
            a_, x_ = mp.mpf(a), mp.mpf(x)
            p = mp.exp(a_ * mp.log(x_) - x_ - mp.loggamma(a_ + 1)) * mp.hyp1f1(
                1, a_ + 1, x_, maxterms=10**8
            )
            cur = mp.log(1 - p)
        if prev is not None and abs(cur - prev) <= mp.mpf("1e-28") * abs(cur):
            return cur
        prev = cur
        dps *= 2
    sys.exit(f"no convergence of the high-precision value at a={a} x={x}")


def large_points() -> list:
    """Points with a huge dof, mostly near the mean (the series is slowest there)."""
    out = []
    for dof in LARGE_DOFS:
        s = math.sqrt(2 * dof)
        xs = {dof + k * s for k in [-8, -3, -1, -0.1, 0, 0.1, 0.5, 1, 2, 3, 5, 8, 12, 20]}
        xs |= {float(dof), float(dof) - 1, dof + 2.0}
        for x in sorted(xs):
            a = dof / 2
            lnq = ln_q_large(a, x / 2)
            nlp = max(-lnq / mp.log(10), mp.mpf(0))
            out.append({"dof": dof, "x": x, "neg_log10_p": mp.nstr(nlp, 30)})
    return out


def gamma_q_points() -> list:
    """ln Q(a, x) for small shapes (where 1 - P cancels) and for huge shapes."""
    pts = []
    for a in [1e-20, 1e-10, 1e-3, 0.1, 0.4]:
        for x in [1e-3, 0.5, 2.0]:
            pts.append((a, x))
    for a in [1e12, 1e9, 2147483647.5]:
        for k in [-3, -1, 0, 1, 3]:
            pts.append((a, a + k * math.sqrt(a)))
    out = []
    for a, x in pts:
        with mp.workdps(60):
            if a < 1:
                lnq = mp.log(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True))
                with mp.workdps(120):
                    lnq2 = mp.log(
                        mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True)
                    )
                if abs(lnq - lnq2) > mp.mpf("1e-28") * abs(lnq2):
                    sys.exit(f"mpmath precision check failed at a={a} x={x}")
            else:
                lnq = ln_q_large(a, x)
        out.append({"a": a, "x": x, "ln_q": mp.nstr(lnq, 30)})
    return out


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
    out += large_points()
    default = Path(__file__).resolve().parent.parent.parent / "tests/fixtures/stats/pvalue_mpmath.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    path.write_text(json.dumps({"mpmath": mp.__version__, "points": out}, indent=0))
    gpath = path.with_name("gamma_q_mpmath.json")
    gpoints = gamma_q_points()
    gpath.write_text(json.dumps({"mpmath": mp.__version__, "points": gpoints}, indent=0))
    print(f"wrote {len(gpoints)} points to {gpath}")
    print(f"wrote {len(out)} points to {path}; max -log10 p = "
          f"{max(float(p['neg_log10_p']) for p in out):.6g}")


if __name__ == "__main__":
    main()
