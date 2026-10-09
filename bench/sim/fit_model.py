#!/usr/bin/env python3
"""Fit the simulation cost model to a sweep TSV (stdlib only).

For each (variant, trace) group, CPU time is modeled as a non-negative linear
combination of:

  1                 fixed cost per run (start-up, closing the FST)      [s]
  cyc               per cycle                                           [ns]
  cyc*N             per register per cycle (eval, and trace compare)    [ns]
  cyc*N*W           per register bit per cycle (eval)                   [ns]
  cyc*N*rho         per register change (trace write; traced groups)   [ns]
  cyc*N*rho*W/2     per toggled bit (trace write; traced groups)        [ns]
  cyc*K             per Python signal touch (variant a)                 [ns]

N*rho is the expected number of register changes per cycle and N*rho*W/2 the
expected number of toggled bits per cycle (an LFSR bit flips with p = 1/2).
Weights are 1/y, so the fit minimizes relative error.
"""
import argparse
import csv
import math
from collections import defaultdict

FEATURES = ["1", "cyc", "cyc*N", "cyc*N*W", "cyc*N*rho", "cyc*N*rho*W/2", "cyc*K"]
UNITS = ["s", "ns", "ns", "ns", "ns", "ns", "ns"]


def feats(N, W, rho, K, cyc):
    return [1.0, cyc, cyc * N, cyc * N * W, cyc * N * rho, cyc * N * rho * W / 2, cyc * K]


def used_features(variant, trace, rows):
    idx = [0, 1, 2, 3]
    if trace == "fst":
        idx += [4, 5]
    if variant == "a" and len({r["K"] for r in rows}) > 1:
        idx += [6]
    return idx


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-300:
            M[c][c] = 1e-300
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def nnls(X, y):
    """Weighted (1/y) non-negative least squares by dropping the most negative coefficient."""
    ncol = len(X[0])
    scale = [max(abs(row[j]) for row in X) or 1.0 for j in range(ncol)]
    Xs = [[row[j] / scale[j] / yi for j in range(ncol)] for row, yi in zip(X, y)]
    ys = [1.0] * len(y)
    active = list(range(ncol))
    while active:
        A = [[sum(r[i] * r[j] for r in Xs) + (1e-12 if i == j else 0.0) for j in active] for i in active]
        b = [sum(r[i] * t for r, t in zip(Xs, ys)) for i in active]
        x = solve(A, b)
        neg = [k for k, v in enumerate(x) if v < 0]
        if not neg:
            coef = [0.0] * ncol
            for k, j in enumerate(active):
                coef[j] = x[k] / scale[j]
            return coef
        active.pop(min(neg, key=lambda k: x[k]))
    return [0.0] * ncol


def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(dict(variant=r["variant"], trace=r["trace"], N=int(r["N"]), W=int(r["W"]),
                             rho=float(r["rho"]), K=int(r["K"]), cyc=int(r["cycles"]),
                             cpu=float(r["cpu_s"]), rss=float(r["rss_mb"]), fst=float(r["fst_mb"]),
                             build=float(r["build_cpu_s"])))
    return rows


class Model:
    def __init__(self, variant, trace, rows):
        self.variant, self.trace = variant, trace
        self.idx = used_features(variant, trace, rows)
        X = [feats(r["N"], r["W"], r["rho"], r["K"], r["cyc"]) for r in rows]
        Xu = [[x[j] for j in self.idx] for x in X]
        c = nnls(Xu, [r["cpu"] for r in rows])
        self.coef = [0.0] * len(FEATURES)
        for j, v in zip(self.idx, c):
            self.coef[j] = v
        pred = [self.predict(r["N"], r["W"], r["rho"], r["K"], r["cyc"]) for r in rows]
        y = [r["cpu"] for r in rows]
        mean = sum(y) / len(y)
        ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
        ss_tot = sum((a - mean) ** 2 for a in y) or 1e-300
        rel = sorted(abs(p - a) / a for a, p in zip(y, pred))
        self.n = len(rows)
        self.r2 = 1 - ss_res / ss_tot
        self.med = rel[len(rel) // 2]
        self.max = rel[-1]

    def terms(self, N, W, rho, K, cyc):
        return [c * f for c, f in zip(self.coef, feats(N, W, rho, K, cyc))]

    def predict(self, N, W, rho, K, cyc):
        return sum(self.terms(N, W, rho, K, cyc))

    def ns_cycle(self, N, W, rho, K):
        """Marginal time per cycle in ns (without the fixed cost)."""
        return sum(self.terms(N, W, rho, K, 1.0)[1:]) * 1e9


def parts(models, v, N, W, rho, K):
    """ns per cycle split into python (per-cycle + touches), design, trace, for variant v."""
    off = models[(v, "off")]
    full = models.get((v, "fst"))
    t = off.terms(N, W, rho, K, 1.0)
    base = t[1] * 1e9
    touch = t[6] * 1e9
    design = (t[2] + t[3]) * 1e9
    trace = (full.ns_cycle(N, W, rho, K) - off.ns_cycle(N, W, rho, K)) if full else 0.0
    return base, touch, design, max(trace, 0.0)


def report(models, rows):
    print("== Fit: CPU seconds = sum_j coef_j * feature_j ==")
    print(f"{'variant':<8}{'trace':<9}{'n':>4}{'R2':>8}{'med.err':>9}{'max.err':>9}   coefficients")
    for (v, tr), m in sorted(models.items()):
        cs = []
        for j, c in enumerate(m.coef):
            if j in m.idx:
                cs.append(f"{FEATURES[j]}={c * (1 if j == 0 else 1e9):.4g}{UNITS[j]}")
        print(f"{v:<8}{tr:<9}{m.n:>4}{m.r2:>8.4f}{m.med:>8.1%}{m.max:>9.1%}   " + "  ".join(cs))
    print("(err = |predicted - measured| / measured, over the points of the group)\n")


def dominance(models):
    print("== Which part dominates (ns per simulated cycle; variant a = cocotb per-cycle, trace = all signals) ==")
    has = lambda v: (v, "off") in models and (v, "fst") in models
    if has("a"):
        print(f"{'N':>6}{'W':>5}{'rho':>5}{'K':>4} | {'py/cycle':>9}{'touches':>9}{'design':>9}{'trace':>9} | dominant")
        for N, W in ((16, 16), (64, 32), (256, 32), (512, 64), (2048, 128)):
            for rho in (0.1, 0.5):
                for K in (0, 16):
                    b, t, d, tr = parts(models, "a", N, W, rho, K)
                    comp = {"python": b + t, "design": d, "trace": tr}
                    print(f"{N:>6}{W:>5}{rho:>5}{K:>4} | {b:>9.0f}{t:>9.0f}{d:>9.0f}{tr:>9.0f} | {max(comp, key=comp.get)}")
        print()
        def rest(N, W, rho):
            b, _, d, tr = parts(models, "a", N, W, rho, 0)
            return b + d + tr
        per_touch = models[("a", "off")].coef[6] * 1e9
        print(f"Python touch cost: {per_touch:.0f} ns per touch.")
        for N, W in ((16, 16), (64, 32), (256, 32), (512, 64)):
            r = rest(N, W, 0.5)
            kstar = r / per_touch if per_touch > 0 else float("inf")
            print(f"  N={N}, W={W}, rho=0.5: touches exceed all other cost at K >= {kstar:.0f} "
                  f"(other cost {r:.0f} ns/cycle)")
        base = parts(models, "a", 16, 16, 0.5, 0)[0]
        print(f"Per-cycle Python overhead (await, no touches): {base:.0f} ns/cycle.")
        for W, rho in ((32, 0.5), (32, 0.1)):
            nstar = None
            for N in range(1, 100000):
                _, _, d, tr = parts(models, "a", N, W, rho, 0)
                if d + tr >= base:
                    nstar = N
                    break
            print(f"  W={W}, rho={rho}: design+trace cost exceeds the per-cycle Python overhead "
                  f"from N >= {nstar} registers (about {4 * nstar + 40 if nstar else '?'} signals)")
        print()
    for v in "bc":
        if has(v):
            print(f"Variant {v} (no per-cycle Python): trace vs design, W=32, rho=0.5")
            for N in (16, 64, 256, 512, 2048):
                _, _, d, tr = parts(models, v, N, 32, 0.5, 0)
                print(f"  N={N:>5}: design {d:>8.0f} ns/cycle, trace {tr:>8.0f} ns/cycle "
                      f"-> {'trace' if tr > d else 'design'} dominates")
    print()


def real_mapping(models, args):
    """Map the real setup on the model. Structural facts only."""
    sig, pts, ppc = args.real_signals, args.real_points, args.points_per_cycle
    cyc = pts / ppc
    N = max(1, round((sig - 40) / 4))
    print("== Real setup mapped onto the model ==")
    print(f"Structural facts: cocotb per-cycle testbench, trace on all signals, {sig} signals, {pts:.3g} time points.")
    print(f"Assumptions: {ppc:g} time points per clock cycle -> {cyc:.3g} cycles; "
          f"{sig} signals ~ {N} model registers (4 handles per register + 40).")
    print(f"Unknown design values are varied: W in {{16,32,64}}, rho in {{0.1,0.5,0.9}}.")
    if not all(k in models for k in (("a", "fst"), ("b", "fst"), ("c", "fst"), ("c", "off"), ("a", "off"))):
        print("(not enough variants in the TSV)")
        return
    rows = []
    for W in (16, 32, 64):
        for rho in (0.1, 0.5, 0.9):
            a0 = models[("a", "fst")].ns_cycle(N, W, rho, 0)
            b = models[("b", "fst")].ns_cycle(N, W, rho, 0)
            c = models[("c", "fst")].ns_cycle(N, W, rho, 0)
            top = models[("b", "fst_top")].ns_cycle(N, W, rho, 0) if ("b", "fst_top") in models else float("nan")
            rows.append((W, rho, a0, b, c, top))
    print(f"{'W':>4}{'rho':>5} | {'a K=0':>8}{'b':>8}{'c':>8}{'b,top':>8} ns/cycle | "
          f"{'a/b':>6}{'a/c':>6}{'b/b,top':>8}  predicted a (K=0), s")
    for W, rho, a0, b, c, top in rows:
        print(f"{W:>4}{rho:>5} | {a0:>8.0f}{b:>8.0f}{c:>8.0f}{top:>8.0f}            | "
              f"{a0 / b:>6.2f}{a0 / c:>6.2f}{b / top:>8.2f}  {a0 * cyc / 1e9:>8.1f}")
    ab = [r[2] / r[3] for r in rows]
    ac = [r[2] / r[4] for r in rows]
    print(f"a -> b (same tracing): factor {min(ab):.2f} to {max(ab):.2f} at K=0 "
          f"(larger with K > 0; cost of one touch per cycle: {models[('a', 'off')].coef[6] * 1e9:.0f} ns)")
    print(f"a -> c (floor, same tracing): factor {min(ac):.2f} to {max(ac):.2f}")
    if args.observed_s:
        obs = args.observed_s * 1e9 / cyc
        fb = [obs / r[3] for r in rows]
        fc = [obs / r[4] for r in rows]
        print(f"If the unexplained part is per-cycle Python, variant b cuts the observed time by "
              f"{min(fb):.1f} to {max(fb):.1f} (W, rho range; W=32, rho=0.5: "
              f"{obs / [r for r in rows if r[0] == 32 and r[1] == 0.5][0][3]:.1f}); variant c by {min(fc):.1f} to {max(fc):.1f}.")
        print("Model error of the traced fits (median / max): " + ", ".join(
            f"{v} {models[(v, 'fst')].med:.0%} / {models[(v, 'fst')].max:.0%}" for v in "abc"))
        print(f"Observed {args.observed_s:g} s per batch = {obs:.0f} ns/cycle.")
        per_touch = models[("a", "off")].coef[6] * 1e9
        for W, rho, a0, *_ in rows:
            if W == 32:
                kk = max(0.0, (obs - a0) / per_touch) if per_touch > 0 else float("nan")
                print(f"  W=32 rho={rho}: model with K=0 gives {a0:.0f} ns/cycle; the rest "
                      f"({max(0.0, obs - a0):.0f} ns/cycle) would be K ~ {kk:.0f} touches per cycle "
                      f"or testbench work the model does not have")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv")
    ap.add_argument("--real-signals", type=int, default=1064)
    ap.add_argument("--real-points", type=float, default=1.48e6, help="time points per batch")
    ap.add_argument("--points-per-cycle", type=float, default=2.0, help="time points per clock cycle")
    ap.add_argument("--observed-s", type=float, default=150.0, help="observed simulation time per batch (s); 0 to skip")
    a = ap.parse_args()
    rows = read(a.tsv)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["trace"])].append(r)
    models = {k: Model(k[0], k[1], v) for k, v in groups.items() if len(v) >= 5}
    report(models, rows)
    dominance(models)
    real_mapping(models, a)


if __name__ == "__main__":
    main()
