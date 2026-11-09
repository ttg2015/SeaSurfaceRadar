"""Resumable, parallel Monte Carlo: Gaussian vs choppy P-M surfaces, HH & VV, 60 & 80 deg.

Design (see REPORT.md for the reasoning):
  * fixed footprint: Thorsos taper g = 55 lambda (12.65 m, 1.11 x Kapp at 80 deg) for every case
  * solve window L = 8 g = 440 lambda (N = 4400 at dx = lambda/10), curvature diagonal on
  * surfaces generated on a 4096-lambda (942 m) periodic domain -> long waves included
  * Gaussian and choppy share coefficients (paired); same normal draws reused across winds
  * one propagator per (surface, pol) serves both incidence angles
  * MOMI with 3 iterations; sigma after iteration 2 is stored too (convergence diagnostic)
  * every realization also gets a bistatic energy check and an edge-leakage check

Output: results/<arm>_U<U>.jsonl, one JSON record per (seed, surface, pol, angle).
Resume: seeds already present in a file are skipped.

usage:  MC_BACKEND=gpu MC_DX_DIV=20 python run_mc.py PLAN [n_workers]     PLAN in {main, quick}
"""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, time
import numpy as np

import surfaces as S
import fastsolve as F
from surfaces import m

LAM = 0.23
K = 2 * np.pi / LAM
G = 55 * LAM
L_WIN_LAM = 440
L_GEN_LAM = {"full": 4096, "periodic": L_WIN_LAM}   # 'periodic' = old design: L_gen = window
ANGLES = (60, 80)
TS = np.linspace(-np.pi / 2, np.pi / 2, 721)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def energy_pred(th, g=G):
    return 1 - (1 + 2 * np.tan(th) ** 2) / (2 * (K * g * np.cos(th)) ** 2)


def solve_realization(U, seed, arm="full", dx_div=10, surfaces=("gauss", "choppy"),
                      direct=False, angles=ANGLES, backend="cpu"):
    """All records for one random sea. direct=True also does a dense solve (validation)."""
    dx = LAM / dx_div
    N = int(round(L_WIN_LAM * dx_div))
    sea = S.Sea(U, seed, LAM, L_GEN_LAM[arm] * LAM)
    recs = []
    for surf in surfaces:
        base = dict(U=U, seed=seed, arm=arm, surf=surf, dx_div=dx_div, backend=backend, g_lam=G / LAM,
                    L_win_lam=L_WIN_LAM, L_gen_lam=L_GEN_LAM[arm],
                    h_rms_nominal=sea.h_rms_nominal, slope_rms_nominal=sea.slope_rms_nominal)
        try:
            w = sea.window(N, dx, surf == "choppy")
        except RuntimeError as e:
            w = None
            err = str(e)
        if w is None or w["jac_min_window"] <= 0:
            # overturned crest inside the window: z(x) is not single-valued, the MFIE in this
            # form cannot be applied. Recorded explicitly (never silently dropped).
            for pol in ("TE", "TM"):
                for a in angles:
                    recs.append(dict(base, pol=pol, theta=a, overturned=True, sigma=None,
                                     jac_min_window=None if w is None else w["jac_min_window"]))
            continue
        diag = {kk: w[kk] for kk in ("jac_min_window", "jac_min_domain", "h_rms_win",
                                     "slope_rms_win", "skew_win")}
        x, z, zx, zxx = w["x"], w["z"], w["zx"], w["zxx"]
        for pol in ("TE", "TM"):
            if backend == "cpu":   # one polarization at a time keeps RAM at one N x N matrix
                A = F.build_A(x, z, zx, zxx, K, pols=(pol,))
            if backend == "gpu":
                import gpusolve as GS
                A = {pol: GS.build_A(x, z, zx, zxx, K, pol)}
            for a in angles:
                th = np.radians(a)
                Ji = m.kirchhoff_current(x, z, zx, K, th, G, pol)
                Js = F.momi(A[pol], Ji, 3) if backend == "cpu" else GS.momi(A[pol], Ji, 3)
                J = Js[-1]
                s3 = m.nrcs(x, z, zx, J, K, th, G, -th, pol)[0]
                s2 = m.nrcs(x, z, zx, Js[-2], K, th, G, -th, pol)[0]
                bis = m.nrcs(x, z, zx, J, K, th, G, TS, pol)
                e = np.trapezoid(bis, TS) / (2 * np.pi * np.cos(th))
                aJ = np.abs(J)
                edge = float(max(aJ[:20].max(), aJ[-20:].max()) / aJ.max())
                r = dict(base, **diag, pol=pol, theta=a, overturned=False, sigma=float(s3),
                         sigma_it2=float(s2), energy=float(e), energy_pred=float(energy_pred(th)),
                         edge_J=edge)
                if direct:
                    Jd = np.linalg.solve(A[pol], Ji) if backend == "cpu" else GS.solve_direct(A[pol], Ji)
                    r["sigma_direct"] = float(m.nrcs(x, z, zx, Jd, K, th, G, -th, pol)[0])
                recs.append(r)
            del A
            if backend == "gpu":
                GS.torch.cuda.empty_cache()
    return recs


BACKEND = os.environ.get("MC_BACKEND", "gpu")
DX_DIV = int(os.environ.get("MC_DX_DIV", "20"))


def _task(args):
    U, seed, arm = args
    t = time.time()
    recs = solve_realization(U, seed, arm, dx_div=DX_DIV, backend=BACKEND,
                             surfaces=("gauss",) if arm == "periodic" else ("gauss", "choppy"))
    return args, recs, time.time() - t


SHARD = int(os.environ.get("MC_SHARD", "0"))
NSHARDS = int(os.environ.get("MC_NSHARDS", "1"))


def fname(arm, U):
    return os.path.join(OUT, f"{arm}_U{U:g}_dx{DX_DIV}_s{SHARD}.jsonl")


def done_seeds(arm, U):
    """Seeds already solved by ANY shard (so the shard layout can change between runs)."""
    import glob as _glob
    s = set()
    for p in _glob.glob(os.path.join(OUT, f"{arm}_U{U:g}_dx{DX_DIV}_s*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    s.add(json.loads(line)["seed"])
                except Exception:
                    pass  # a partially written last line from an interruption
    return s


PLANS = {
    # (U, arm, n_seeds) in blocks, so every case grows together and partial results are usable
    "main": [(10.0, "full", 2400), (5.0, "full", 1600), (15.0, "full", 1600), (15.0, "periodic", 800)],
    "quick": [(10.0, "full", 32)],
    # tail sweep: 10,000 seas per wind (P ~ 1e-3 with ~+-50 % CI)
    "big": [(10.0, "full", 10000), (5.0, "full", 10000), (15.0, "full", 10000)],
}


def tasks_for(plan, block=200):
    spec = PLANS[plan]
    done = {(U, arm): done_seeds(arm, U) for U, arm, _ in spec}
    out, b = [], 0
    while True:
        added = False
        for U, arm, n in spec:
            frac = n / max(n for *_, n in spec)
            lo, hi = int(b * block * frac), int((b + 1) * block * frac)
            hi = min(hi, n)
            for s in range(lo, hi):
                added = True
                if s not in done[(U, arm)]:
                    out.append((U, s, arm))
        if not added:
            return out
        b += 1


if __name__ == "__main__":
    # one process per shard (multiprocessing.Pool cannot spawn in this environment):
    #   MC_SHARD=i MC_NSHARDS=n python run_mc.py main
    plan = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    # weighted split: this shard owns seed slots [MC_SLOT_LO, MC_SLOT_HI) of MC_NSLOTS
    nslots = int(os.environ.get("MC_NSLOTS", NSHARDS))
    lo = int(os.environ.get("MC_SLOT_LO", SHARD))
    hi = int(os.environ.get("MC_SLOT_HI", SHARD + 1))
    tasks = [t for t in tasks_for(plan) if lo <= t[1] % nslots < hi]
    print(f"shard {SHARD}/{NSHARDS}: {len(tasks)} realizations, backend={BACKEND}, dx=lambda/{DX_DIV}", flush=True)
    files = {}
    t0 = time.time()
    paused = 0.0
    STOP, PAUSE = os.path.join(OUT, "STOP"), os.path.join(OUT, "PAUSE")
    for n, (U, seed, arm) in enumerate(tasks, 1):
        # control files, checked between realizations (see sweep.sh): STOP = exit cleanly, PAUSE = hold
        if os.path.exists(STOP):
            print(f"[shard {SHARD}] STOP file found: exiting cleanly after {n-1} realizations", flush=True)
            break
        if os.path.exists(PAUSE):
            print(f"[shard {SHARD}] paused", flush=True)
            tp = time.time()
            if BACKEND == "gpu":
                import gpusolve as _GS
                _GS.torch.cuda.empty_cache()
            while os.path.exists(PAUSE) and not os.path.exists(STOP):
                time.sleep(5)
            paused += time.time() - tp
            if os.path.exists(STOP):
                print(f"[shard {SHARD}] STOP file found while paused: exiting", flush=True)
                break
            print(f"[shard {SHARD}] resumed", flush=True)
        _, recs, dt = _task((U, seed, arm))
        key = (arm, U)
        if key not in files:
            files[key] = open(fname(arm, U), "a")
        for r in recs:
            files[key].write(json.dumps(r) + "\n")
        files[key].flush()
        if n % 20 == 0 or n == len(tasks):
            el = time.time() - t0 - paused
            print(f"[shard {SHARD}] {n}/{len(tasks)}  {el/60:.1f} min, ETA {el/n*(len(tasks)-n)/60:.1f} min "
                  f"(last {dt:.1f}s)", flush=True)
    for f in files.values():
        f.close()
    else:
        print(f"[shard {SHARD}] done", flush=True)
