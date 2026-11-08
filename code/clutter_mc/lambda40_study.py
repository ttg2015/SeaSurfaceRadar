"""Is the choppy-surface HH tail real or a crest-resolution artifact?

Re-solve CHOPPY surfaces at dx = lambda/40 (GPU) for
  (a) the top-8 HH spikes at each (U, angle) in the lambda/20 production run, and
  (b) a random subset of seeds at U = 10 and 15 m/s (unbiased: estimates the lambda/20 bias on the
      mean level and on m2, paired seed-by-seed with the lambda/20 values).
Also records MOMI convergence at lambda/40 and the largest crest curvature in the window.
Writes results/lambda40.jsonl (resumable)."""
import json, glob, os, sys
import numpy as np
import run_mc as R

N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 60
HERE = os.path.dirname(os.path.abspath(__file__))
data = {}
for p in glob.glob(os.path.join(R.OUT, "full_U*_dx20_s*.jsonl")):
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        if r["surf"] == "choppy":
            data.setdefault((r["U"], r["pol"], r["theta"]), {})[r["seed"]] = r["sigma"]

todo = []
for U in (10.0, 15.0, 5.0):
    for th in (80, 60):
        d = data[(U, "TE", th)]
        seeds = np.array(sorted(d)); s = np.array([d[q] for q in seeds])
        todo += [(U, int(q), "spike") for q in seeds[np.argsort(-s)[:8]]]
rng = np.random.default_rng(40)
for U in (10.0, 15.0):
    seeds = np.array(sorted(data[(U, "TE", 80)]))
    todo += [(U, int(q), "random") for q in rng.choice(seeds, N_RANDOM, replace=False)]
# random first (unbiased estimate is the priority), then spikes; de-duplicate
order = [t for t in todo if t[2] == "random"] + [t for t in todo if t[2] == "spike"]
seen, todo = set(), []
for U, s, why in order:
    if (U, s) not in seen:
        seen.add((U, s)); todo.append((U, s, why))

outp = os.path.join(R.OUT, "lambda40.jsonl")
done = set()
if os.path.exists(outp):
    for l in open(outp):
        r = json.loads(l); done.add((r["U"], r["seed"]))
print(f"{len(todo)} choppy realizations, {len(done)} already done", flush=True)
with open(outp, "a") as f:
    for U, seed, why in todo:
        if (U, seed) in done:
            continue
        for r in R.solve_realization(U, seed, dx_div=40, surfaces=("choppy",), backend="gpu"):
            r["why"] = why
            r["sigma20"] = data[(U, r["pol"], r["theta"])][seed]
            f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"U={U} seed={seed} ({why}) done", flush=True)
