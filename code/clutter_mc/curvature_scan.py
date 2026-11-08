"""For every choppy realization in the production run: largest crest curvature (and smallest
dx/du) inside the illuminated region, at the production sampling. Writes results/curvature.jsonl."""
import json, glob, os
import numpy as np
import surfaces as S, run_mc as R
seen = set()
for p in glob.glob(os.path.join(R.OUT, "full_U*_dx20_s*.jsonl")):
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        seen.add((r["U"], r["seed"]))
outp = os.path.join(R.OUT, "curvature.jsonl")
have = set()
if os.path.exists(outp):
    for l in open(outp):
        r = json.loads(l); have.add((r["U"], r["seed"]))
with open(outp, "a") as f:
    for U, seed in sorted(seen - have):
        sea = S.Sea(U, seed, R.LAM, 4096 * R.LAM)
        rec = dict(U=U, seed=seed)
        for surf in ("gauss", "choppy"):
            w = sea.window(8800, R.LAM / 20, surf == "choppy")
            kap = np.abs(w["zxx"]) / (1 + w["zx"] ** 2) ** 1.5
            for a in (60, 80):
                lit = np.abs(w["x"] + w["z"] * np.tan(np.radians(a))) < 2 * R.G
                rec[f"{surf}_kmax_{a}"] = float(kap[lit].max())
        f.write(json.dumps(rec) + "\n")
