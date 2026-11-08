"""Spot checks on a few realizations before trusting production numbers:
  1) sampling convergence dx = lambda/10 vs lambda/20 (choppy adds harmonics above Kc)
  2) MOMI(3) vs dense direct solve
  3) energy conservation vs Thorsos finite-beam prediction
  4) edge leakage of the current
Uses multithreaded BLAS (run alone, not alongside the production run)."""
import json, os, sys
import numpy as np
import run_mc as R

out = os.path.join(R.OUT, "spotcheck.jsonl")
os.makedirs(R.OUT, exist_ok=True)
cases = [(10.0, s) for s in (9001, 9002, 9003)] + [(15.0, s) for s in (9001, 9002, 9003)]
with open(out, "w") as f:
    for U, seed in cases:
        res = {}
        for dd in (10, 20):
            for r in R.solve_realization(U, seed, dx_div=dd, direct=(dd == 10)):
                res[(dd, r["surf"], r["pol"], r["theta"])] = r
                f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"U={U} seed={seed}")
        for surf in ("gauss", "choppy"):
            for pol in ("TE", "TM"):
                for a in R.ANGLES:
                    a10, a20 = res[(10, surf, pol, a)], res[(20, surf, pol, a)]
                    if a10["overturned"]:
                        print(f"  {surf:6s} {pol} {a}: OVERTURNED")
                        continue
                    d_dx = 10 * np.log10(a20["sigma"] / a10["sigma"])
                    d_dir = 10 * np.log10(a10["sigma"] / a10["sigma_direct"])
                    d_it = 10 * np.log10(a10["sigma"] / a10["sigma_it2"])
                    print(f"  {surf:6s} {pol} {a}: sigma {10*np.log10(a10['sigma']):7.2f} dB | "
                          f"dx/20-dx/10 {d_dx:+.3f} dB | MOMI3-direct {d_dir:+.1e} dB | it3-it2 {d_it:+.1e} dB | "
                          f"energy {a10['energy']:.4f} (pred {a10['energy_pred']:.4f}) | edge {a10['edge_J']:.1e} | "
                          f"min dx/du {a10['jac_min_window']:.2f}", flush=True)
