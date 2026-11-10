"""Preliminary look at the running 10k sweep without touching the report outputs.

Runs analyze.py with its outputs redirected to results/summary_prelim.txt and figures/prelim/, then a
tail-robustness table: the new seeds have NOT had their spikes re-solved at lambda/40 yet, and lambda/20
cusp artifacts live exactly in the tail, so m2 is shown with the top 1/3/10 values removed and the top
spikes are listed with their MOMI and energy diagnostics.
"""
import os
import numpy as np
import analyze as A

A.TAG = "_prelim"
A.FIG = os.path.join(A.HERE, "figures", "prelim")
os.makedirs(A.FIG, exist_ok=True)
A.main()

data = A.load()
print("\nTAIL ROBUSTNESS (full arm, 80 deg): m2 with the k largest sigma removed")
print(f"{'case':26s} {'n':>5s} {'m2':>6s} {'-1':>6s} {'-3':>6s} {'-10':>6s}   top spikes: seed, sigma/<sigma>, it3-vs-it2 dB, energy err, l40?")
for U in (5.0, 10.0, 15.0):
    for pol in ("TE", "TM"):
        for surf in ("gauss", "choppy"):
            recs = data.get(("full", U, surf, pol, 80), {})
            rr = [r for r in recs.values() if not r["overturned"]]
            if not rr:
                continue
            s = np.array([r["sigma"] for r in rr])
            o = np.argsort(s)[::-1]
            m2k = [A.m2(np.delete(s, o[:k])) if k else A.m2(s) for k in (0, 1, 3, 10)]
            top = " | ".join(
                f"{rr[i]['seed']}: {s[i]/s.mean():5.1f}x {abs(10*np.log10(rr[i]['sigma']/rr[i]['sigma_it2'])):.0e} "
                f"{rr[i]['energy']/rr[i]['energy_pred']-1:+.0e}{' L40' if rr[i].get('l40') else ''}" for i in o[:3])
            print(f"U={U:4.1f} {surf:6s} {A.POLNAME[pol]} 80°  {len(s):5d} " + " ".join(f"{v:6.2f}" for v in m2k) + "   " + top)
