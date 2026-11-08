"""Summarise results/lambda40.jsonl.
  random subset: lambda/20 vs lambda/40 for the same choppy seeds (bias in mean, m2), and the paired
                 choppy(lambda/40) - Gaussian(lambda/20) differences on those seeds
  spikes:        lambda/20 vs lambda/40 for the largest choppy HH values."""
import json, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(7)
B = 4000
m2 = lambda s: np.mean(s**2) / np.mean(s) ** 2
P = {"TE": "HH", "TM": "VV"}

gauss = {}
for p in glob.glob(os.path.join(RES, "full_U*_dx20_s*.jsonl")):
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        if r["surf"] == "gauss":
            gauss[(r["U"], r["seed"], r["pol"], r["theta"])] = r["sigma"]
L = [json.loads(l) for l in open(os.path.join(RES, "lambda40.jsonl"))]
lines = []
pr = lambda s: (print(s), lines.append(s))

pr("RANDOM SUBSET (unbiased): same choppy seeds at lambda/20 and lambda/40; Gaussian at lambda/20")
pr(f"{'case':16s} {'n':>3s} {'per-real |dB| med/max':>22s} {'mean bias dB [CI]':>24s} {'m2 l/20':>8s} {'m2 l/40':>8s} "
   f"{'d<s0> chop40-gauss [CI]':>26s} {'d m2 chop40-gauss [CI]':>26s} {'(same with l/20)':>18s} {'MOMI>0.01dB':>11s}")
for U in (10.0, 15.0):
    for pol in ("TE", "TM"):
        for th in (60, 80):
            rs = [r for r in L if r["why"] == "random" and r["U"] == U and r["pol"] == pol and r["theta"] == th]
            if not rs:
                continue
            s40 = np.array([r["sigma"] for r in rs]); s20 = np.array([r["sigma20"] for r in rs])
            sg = np.array([gauss[(U, r["seed"], pol, th)] for r in rs])
            nconv = sum(abs(10 * np.log10(r["sigma"] / r["sigma_it2"])) > 0.01 for r in rs)
            d = 10 * np.log10(s40 / s20)
            n = len(rs); idx = rng.integers(0, n, (B, n))
            bias = np.array([10 * np.log10(s40[i].mean() / s20[i].mean()) for i in idx])
            dm = np.array([10 * np.log10(s40[i].mean() / sg[i].mean()) for i in idx])
            dm2 = np.array([m2(s40[i]) - m2(sg[i]) for i in idx])
            ci = lambda a: np.percentile(a, [2.5, 97.5])
            b, c1, c2 = ci(bias), ci(dm), ci(dm2)
            pr(f"U={U:4.1f} {P[pol]} {th}°   {n:3d} {np.median(abs(d)):10.2f}/{abs(d).max():6.2f}   "
               f"{10*np.log10(s40.mean()/s20.mean()):+6.2f} [{b[0]:+5.2f},{b[1]:+5.2f}] {m2(s20):8.2f} {m2(s40):8.2f} "
               f"{10*np.log10(s40.mean()/sg.mean()):+7.2f} [{c1[0]:+5.2f},{c1[1]:+5.2f}] {m2(s40)-m2(sg):+7.2f} [{c2[0]:+5.2f},{c2[1]:+5.2f}] "
               f"{10*np.log10(s20.mean()/sg.mean()):+6.2f}/{m2(s20)-m2(sg):+5.2f} {nconv:8d}")

pr("\nSPIKES: largest choppy HH values at lambda/20, re-solved at lambda/40")
for r in sorted([r for r in L if r["why"] == "spike" and r["pol"] == "TE"], key=lambda r: (r["U"], r["theta"], -r["sigma20"])):
    pr(f"  U={r['U']:4.1f} seed {r['seed']:5d} HH {r['theta']}°: lambda/20 {10*np.log10(r['sigma20']):7.2f} dB -> "
       f"lambda/40 {10*np.log10(r['sigma']):7.2f} dB ({10*np.log10(r['sigma']/r['sigma20']):+6.2f}); "
       f"MOMI it3-it2 {10*np.log10(r['sigma']/r['sigma_it2']):+.1e} dB; energy {r['energy']:.3f}")
open(os.path.join(RES, "lambda40_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
