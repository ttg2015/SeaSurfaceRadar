"""Sampling convergence: same realizations at dx = lambda/10, /20, /40 (GPU), Gaussian and
choppy. Decides the production dx. Writes results/convergence.jsonl and prints a summary."""
import json, os, sys
import numpy as np
import run_mc as R

out = os.path.join(R.OUT, "convergence.jsonl")
seeds = range(8001, 8007)
with open(out, "a") as f:
    for U in (10.0, 15.0, 5.0):
        for seed in seeds:
            for dd in (10, 20, 40):
                for r in R.solve_realization(U, seed, dx_div=dd, backend="gpu"):
                    f.write(json.dumps(r) + "\n")
                f.flush()
            print(f"U={U} seed={seed} done", flush=True)
