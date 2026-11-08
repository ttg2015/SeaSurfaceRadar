"""MOMI safety net: re-solve with a dense direct solve every realization whose MOMI iterations
2 and 3 differ by more than THRESH dB in any case. Results go to results/fixups.jsonl; analyze.py
substitutes sigma_direct for those records and reports how many were substituted.
usage: python fixup.py THRESH_dB BACKEND"""
import json, glob, os, sys
import numpy as np
import run_mc as R

thresh, backend = float(sys.argv[1]), sys.argv[2]
outp = os.path.join(R.OUT, "fixups.jsonl")
done = set()
if os.path.exists(outp):
    for l in open(outp):
        r = json.loads(l); done.add((r["arm"], r["U"], r["seed"]))
todo = set()
for p in glob.glob(os.path.join(R.OUT, "*_dx20_s*.jsonl")):
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        if r["sigma"] and abs(10 * np.log10(r["sigma"] / r["sigma_it2"])) > thresh:
            todo.add((r["arm"], r["U"], r["seed"]))
todo -= done
print(f"{len(todo)} realizations to re-solve directly", flush=True)
with open(outp, "a") as f:
    for arm, U, seed in sorted(todo):
        surfs = ("gauss",) if arm == "periodic" else ("gauss", "choppy")
        for r in R.solve_realization(U, seed, arm, dx_div=20, surfaces=surfs, direct=True, backend=backend):
            f.write(json.dumps(r) + "\n")
            print(f"  {arm} U={U} seed={seed} {r['surf']} {r['pol']} {r['theta']}: MOMI {10*np.log10(r['sigma']):.2f} dB, "
                  f"direct {10*np.log10(r['sigma_direct']):.2f} dB", flush=True)
        f.flush()
