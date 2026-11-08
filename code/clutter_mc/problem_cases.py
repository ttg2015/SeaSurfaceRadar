"""Figures of the problem cases:
  fig6_cusp_case.png        seed 689 (U=10, choppy): the near-cusp, sampling at lambda/20 vs /40, the
                            HH current at both resolutions, MOMI error vs iteration
  fig7_curvature_vs_sigma   sigma/<sigma> vs crest sharpness kappa_max*dx (lambda/20), choppy HH
  fig8_energy_tilt          energy-check deficit vs long-wave tilt under the footprint (U=15, HH 80)
                            and the bistatic pattern of the worst case vs a normal one
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import surfaces as S, fastsolve as F, run_mc as R
from surfaces import m

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
th80 = np.radians(80)


def load(pattern):
    out = {}
    for p in glob.glob(os.path.join(R.OUT, pattern)):
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            out[(r["U"], r["seed"], r["surf"], r["pol"], r["theta"])] = r
    return out


# ------------------------------------------------------------------ fig 6: the cusp case
U, seed = 10.0, 689
sea = S.Sea(U, seed, R.LAM, 4096 * R.LAM)
res = {}
for dd in (20, 40):
    w = sea.window(440 * dd, R.LAM / dd, True)
    x, z, zx, zxx = w["x"], w["z"], w["zx"], w["zxx"]
    A = F.build_A(x, z, zx, zxx, R.K, pols=("TE",))["TE"]
    Ji = m.kirchhoff_current(x, z, zx, R.K, th80, R.G, "TE")
    Js = F.momi(A, Ji, 8)
    Jref = np.linalg.solve(A, Ji) if dd == 20 else Js[-1]
    err = [np.linalg.norm(j - Jref) / np.linalg.norm(Jref) for j in Js]
    sig = m.nrcs(x, z, zx, Jref, R.K, th80, R.G, -th80, "TE")[0]
    res[dd] = dict(x=x, z=z, zx=zx, zxx=zxx, J=Jref, err=err, sig=sig)
    del A
    print(f"lambda/{dd}: sigma HH80 = {10*np.log10(sig):.2f} dB")
wg = sea.window(8800, R.LAM / 20, False)
r20, r40 = res[20], res[40]
kap = np.abs(r40["zxx"]) / (1 + r40["zx"] ** 2) ** 1.5
ic = np.argmax(kap)
xc = r40["x"][ic]

fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
a = ax[0, 0]
a.plot(wg["x"], wg["z"], "C0", lw=0.8, label="Gaussian (same seed)")
a.plot(r40["x"], r40["z"], "C3", lw=0.8, label="choppy")
a.axvspan(-R.G, R.G, color="gold", alpha=0.15, label="footprint ±g")
a.plot(xc, r40["z"][ic], "kv", ms=10, label=f"sharpest crest: radius {1000/kap[ic]:.1f} mm")
a.set(xlabel="x [m]", ylabel="z [m]", title=f"U = 10 m/s, seed {seed}: the window the solver sees")
a.legend(fontsize=8)
a = ax[0, 1]
sel40 = np.abs(r40["x"] - xc) < 0.08
sel20 = np.abs(r20["x"] - xc) < 0.08
xf = np.linspace(xc - 0.08, xc + 0.08, 4001)
wf = sea.window(4001, xf[1] - xf[0], True)  # not aligned; just for a smooth reference curve below
uu = np.linspace(-0.3, 0.3, 20001) + xc
a.plot(r40["x"][sel40] * 100, r40["z"][sel40] * 100, "C3.-", lw=0.8, ms=5, label="λ/40 samples (5.75 mm)")
a.plot(r20["x"][sel20] * 100, r20["z"][sel20] * 100, "ks", mfc="none", ms=8, label="λ/20 samples (11.5 mm)")
a.set(xlabel="x [cm]", ylabel="z [cm]", title="Zoom ±8 cm on the crest: λ/20 straddles it with ~2 points")
a.set_aspect("equal")
a.legend(fontsize=8)
a = ax[1, 0]
a.semilogy(r20["x"], np.abs(r20["J"]), "k", lw=0.7, label=f"λ/20 direct solve: σ_HH(80°) = {10*np.log10(r20['sig']):.1f} dB")
a.semilogy(r40["x"], np.abs(r40["J"]), "C3", lw=0.7, alpha=0.8, label=f"λ/40: σ_HH(80°) = {10*np.log10(r40['sig']):.1f} dB")
a.axvline(xc, color="k", ls=":", lw=1)
a.set(xlabel="x [m]", ylabel="|J| (HH, 80°)", title="Induced current: λ/20 has a spurious resonance at the crest")
a.legend(fontsize=8)
a = ax[1, 1]
a.semilogy(range(len(r20["err"])), r20["err"], "ko-", label="λ/20: error grows ×2 per iteration (diverges)")
a.semilogy(range(len(r40["err"])), r40["err"], "C3o-", label="λ/40: converges to round-off")
a.set(xlabel="MOMI iteration (0 = Born term)", ylabel="‖J_MOMI − J_direct‖ / ‖J_direct‖",
      title="MOMI divergence is the symptom of an under-resolved crest")
a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_cusp_case.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------------------ fig 7: sharpness vs sigma
cur = {(r["U"], r["seed"]): r for r in map(json.loads, open(os.path.join(R.OUT, "curvature.jsonl")))}
dat = load("full_U*_dx20_s*.jsonl")
fix = {(r["U"], r["seed"], r["surf"], r["pol"], r["theta"]): r["sigma_direct"]
       for r in map(json.loads, open(os.path.join(R.OUT, "fixups.jsonl")))}
dx = R.LAM / 20
fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for j, Uv in enumerate((5.0, 10.0, 15.0)):
    for i, th in enumerate((60, 80)):
        a = ax[i, j]
        for surf, c in (("gauss", "C0"), ("choppy", "C3")):
            ks = [k for k in cur if k[0] == Uv and (Uv, k[1], surf, "TE", th) in dat]
            s = np.array([fix.get((Uv, q, surf, "TE", th), dat[(Uv, q, surf, "TE", th)]["sigma"]) for _, q in ks])
            kd = np.array([cur[k][f"{surf}_kmax_{th}"] * dx for k in ks])
            a.semilogy(kd, s / s.mean(), ".", color=c, ms=3, alpha=0.5, label=f"{surf} (n={len(s)})")
            if surf == "choppy" and (Uv, 689) in cur and Uv == 10.0:
                raw = dat[(Uv, 689, surf, "TE", th)]["sigma"]
                a.semilogy(cur[(Uv, 689)][f"choppy_kmax_{th}"] * dx, raw / s.mean(), "kx", ms=10, mew=2,
                           label="seed 689 before fix (MOMI diverged)")
        a.axvline(1, color="k", ls=":", lw=1)
        a.set(title=f"U = {Uv:g} m/s, HH {th}°", xlabel="sharpest crest in footprint: κ_max·dx  (λ/20)")
        if j == 0: a.set_ylabel("σ / ⟨σ⟩")
        a.legend(fontsize=7)
fig.suptitle("Is the HH tail made of sharp crests? Right of the dotted line, the crest radius is smaller than the grid step.")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_curvature_vs_sigma.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------------------ fig 8: energy check vs tilt
tilt, dev, seeds = [], [], []
for (Uv, q, surf, pol, th), r in dat.items():
    if Uv == 15.0 and surf == "gauss" and pol == "TE" and th == 80:
        w = S.Sea(15.0, q, R.LAM, 4096 * R.LAM).window(4400, R.LAM / 10, False)
        a_ = w["x"] + w["z"] * np.tan(th80)
        lit = np.exp(-2 * a_**2 / R.G**2) > 0.1
        tilt.append(np.polyfit(w["x"][lit], w["z"][lit], 1)[0]); dev.append(r["energy"] / r["energy_pred"] - 1)
        seeds.append(q)
tilt, dev = np.array(tilt), np.array(dev)
worst, normal = seeds[int(np.argmin(dev))], seeds[int(np.argmin(np.abs(tilt)))]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
a = ax[0]
a.plot(tilt, 100 * dev, ".", ms=4, alpha=0.6)
a.axvline(-np.tan(np.radians(5)), color="C3", ls="--", label="tilt = −tan 5°: specular lobe reaches the horizon")
a.axhspan(-2, 2, color="0.85", label="±2 % band")
a.set(xlabel="mean slope of the long wave under the footprint (negative = facing away from radar)",
      ylabel="energy check deviation [%]", title=f"U = 15 m/s, Gaussian, HH 80° (n={len(dev)})")
a.legend(fontsize=8)
a = ax[1]
for q, c, lbl in ((normal, "C0", "flat footprint"), (worst, "C3", "worst energy deficit")):
    w = S.Sea(15.0, q, R.LAM, 4096 * R.LAM).window(8800, R.LAM / 20, False)
    x, z, zx, zxx = w["x"], w["z"], w["zx"], w["zxx"]
    A = F.build_A(x, z, zx, zxx, R.K, pols=("TE",))["TE"]
    J = F.momi(A, m.kirchhoff_current(x, z, zx, R.K, th80, R.G, "TE"), 3)[-1]
    del A
    ts = np.linspace(-np.pi / 2, np.pi / 2, 1441)
    b = m.nrcs(x, z, zx, J, R.K, th80, R.G, ts, "TE")
    i = seeds.index(q)
    a.semilogy(np.degrees(ts), b, c, lw=0.9, label=f"seed {q}: {lbl} (tilt {tilt[i]:+.3f}, energy {100*dev[i]:+.1f} %)")
a.axvline(80, color="k", ls=":", lw=1); a.axvline(-80, color="k", ls=":", lw=1)
a.set(xlabel="scattering angle θs [deg] (+80 = flat-surface specular, −80 = backscatter)", ylabel="σ⁰(θs)",
      ylim=(1e-7, None), title="Bistatic pattern: the tilted patch throws its specular lobe past 90°")
a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_energy_tilt.png"), dpi=130); plt.close(fig)
print("done")
