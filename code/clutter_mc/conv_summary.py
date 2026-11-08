"""Summarise results/convergence.jsonl: per-realization |dB| change lambda/10->/20 and /20->/40."""
import json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = {}
for line in open(os.path.join(HERE, "results", "convergence.jsonl")):
    r = json.loads(line)
    if r["sigma"] is not None:
        d[(r["U"], r["seed"], r["surf"], r["pol"], r["theta"], r["dx_div"])] = r["sigma"]
groups = defaultdict(lambda: ([], []))
for (U, seed, surf, pol, th, dd), s in d.items():
    if dd != 20:
        continue
    k = (surf, {"TE": "HH", "TM": "VV"}[pol], th, U)
    if (U, seed, surf, pol, th, 10) in d:
        groups[k][0].append(10 * np.log10(s / d[(U, seed, surf, pol, th, 10)]))
    if (U, seed, surf, pol, th, 40) in d:
        groups[k][1].append(10 * np.log10(d[(U, seed, surf, pol, th, 40)] / s))
print(f"{'case':28s} {'n':>3s} | {'/10->/20 median|dB|':>20s} {'max|dB|':>8s} | {'/20->/40 median|dB|':>20s} {'max|dB|':>8s}")
for k in sorted(groups):
    a, b = (np.abs(np.array(v)) for v in groups[k])
    print(f"{k[0]:6s} {k[1]} {k[2]}° U={k[3]:4.1f}      {len(b):3d} | {np.median(a):20.3f} {a.max():8.3f} | "
          f"{np.median(b) if len(b) else np.nan:20.3f} {b.max() if len(b) else np.nan:8.3f}")
