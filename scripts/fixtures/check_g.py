"""Compute the G statistic of one fixture table with mpmath (60 digits).

Run it from the repository root, in a virtual environment with mpmath 1.4.1:

    VENV/bin/python -I scripts/fixtures/check_g.py TABLE_NAME

TABLE_NAME is the `name` of a case in `tests/fixtures/stats/scipy_tables.json`.
"""
import json, sys
from pathlib import Path
import mpmath as mp

name = sys.argv[1]
d = json.loads((Path(__file__).resolve().parent.parent.parent / "tests/fixtures/stats/scipy_tables.json").read_text())
case = next(c for c in d["cases"] if c["name"] == name)
mp.mp.dps = 60
t = [[mp.mpf(x) for x in r] for r in case["table"]]
rows = [r for r in t if sum(r) > 0]
ncols = len(rows[0])
cols = [j for j in range(ncols) if sum(r[j] for r in rows) > 0]
rows = [[r[j] for j in cols] for r in rows]
R = [sum(r) for r in rows]; C = [sum(r[j] for r in rows) for j in range(len(cols))]; N = sum(R)
g = mp.mpf(0)
for i, r in enumerate(rows):
    for j, f in enumerate(r):
        if f > 0:
            g += 2 * f * mp.log(f * N / (R[i] * C[j]))
print("mpmath", mp.nstr(g, 20), " scipy", repr(case["g"]["stat"]))
