"""Analysis of results/*.jsonl -> figures/ and results/summary.txt.

Statistics (all with a nonparametric bootstrap over realizations, B = 2000, percentile 95% CIs):
  * mean backscatter <sigma0> in dB, compared with first-order Bragg (SPM):
        HH: (pi alpha / 4) cos^4 th / sin^3 th      VV: (pi alpha / 4)(1 + sin^2 th)^2 / sin^3 th
  * m2 = <sigma^2>/<sigma>^2 (2 for exponential speckle; K-distribution: 2 (1 + 1/nu))
  * exceedance P(sigma/<sigma> > tau) with a CI band
  * PAIRED choppy-vs-Gaussian differences: the same seeds give both surfaces, so we resample
    seeds jointly. This is much tighter than comparing two independent CIs.
"""
import json, os, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
ALPHA = 8.1e-3
B = 2000
TAU = np.linspace(0, 20, 401)
rng = np.random.default_rng(12345)
POLNAME = {"TE": "HH", "TM": "VV"}
PATCH40 = os.environ.get("MC_PATCH40", "1") == "1"
TAG = "" if PATCH40 else "_raw20"


def bragg(pol, a):
    th = np.radians(a)
    f = np.cos(th) ** 4 if pol == "TE" else (1 + np.sin(th) ** 2) ** 2
    return np.pi * ALPHA / 4 * f / np.sin(th) ** 3


def load():
    data = defaultdict(dict)   # (arm,U,surf,pol,theta) -> {seed: rec}
    for p in glob.glob(os.path.join(RES, f"*_U*_dx{os.environ.get('MC_DX_DIV', '20')}_s*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                data[(r["arm"], r["U"], r["surf"], r["pol"], r["theta"])][r["seed"]] = r
    # MOMI safety net: realizations re-solved with a dense direct solve (fixup.py)
    fx = os.path.join(RES, "fixups.jsonl")
    nfix = 0
    if os.path.exists(fx):
        for line in open(fx):
            r = json.loads(line)
            key = (r["arm"], r["U"], r["surf"], r["pol"], r["theta"])
            if r["seed"] in data.get(key, {}):
                old = data[key][r["seed"]]
                if abs(10 * np.log10(old["sigma"] / r["sigma_direct"])) > 0.01:
                    nfix += 1
                    print(f"  [fixup] {key} seed {r['seed']}: MOMI {10*np.log10(old['sigma']):.2f} dB -> "
                          f"direct {10*np.log10(r['sigma_direct']):.2f} dB")
                data[key][r["seed"]] = dict(old, sigma=r["sigma_direct"], sigma_it2=r["sigma_direct"], fixed=True)
    print(f"  {nfix} records replaced by direct solves")
    # lambda/40 re-solves (lambda40_study.py): top spikes + a random subset of choppy seeds
    l40 = os.path.join(RES, "lambda40.jsonl")
    if PATCH40 and os.path.exists(l40):
        n40 = 0
        for line in open(l40):
            r = json.loads(line)
            key = ("full", r["U"], r["surf"], r["pol"], r["theta"])
            if r["seed"] in data.get(key, {}) and r["sigma"] is not None:
                data[key][r["seed"]] = dict(data[key][r["seed"]], sigma=r["sigma"], sigma_it2=r["sigma_it2"],
                                            energy=r["energy"], l40=True)
                n40 += 1
        print(f"  {n40} choppy records replaced by lambda/40 re-solves")
    return data


def exceed(s, tau=TAU):
    v = np.sort(s / s.mean())
    return 1 - np.searchsorted(v, tau, side="right") / len(v)


def boot(s, fn, idx):
    return np.array([fn(s[i]) for i in idx])


def m2(s):
    return np.mean(s**2) / np.mean(s) ** 2


def ci(a, axis=0):
    return np.nanpercentile(a, [2.5, 97.5], axis=axis)


def trust_depth(s, idx):
    """Smallest P(sigma/<sigma> > tau) still estimated to within a factor ~1.5 (95% CI
    half-width in log10 < 0.18), and the count-based rule of thumb 10/n."""
    E = np.array([exceed(s[i]) for i in idx])
    p = exceed(s)
    lo, hi = ci(E)
    with np.errstate(divide="ignore", invalid="ignore"):
        good = (lo > 0) & (np.log10(hi / lo) / 2 < 0.18)
    pmin = p[good].min() if good.any() else np.nan
    return pmin, 10 / len(s)


def main():
    data = load()
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))
    keys = sorted(data)
    stats = {}
    # ---------------- per-case statistics -------------------------------------------------
    P("=" * 100)
    P(f"{'case':38s} {'n':>5s} {'ovt':>4s} {'<s0> dB [95% CI]':>24s} {'Bragg':>7s} {'m2 [95% CI]':>22s} "
      f"{'nu_K':>6s} {'P-trust':>8s}")
    for key in keys:
        arm, U, surf, pol, a = key
        recs = data[key]
        seeds = np.array(sorted(recs))
        ov = sum(recs[s]["overturned"] for s in seeds)
        s = np.array([recs[q]["sigma"] for q in seeds if not recs[q]["overturned"]], float)
        n = len(s)
        idx = rng.integers(0, n, (B, n))
        mb = boot(s, np.mean, idx)
        m2b = boot(s, m2, idx)
        pt, p10 = trust_depth(s, idx)
        mu = 10 * np.log10(s.mean())
        mlo, mhi = 10 * np.log10(ci(mb))
        r2 = m2(s)
        r2lo, r2hi = ci(m2b)
        nu = 2 / (r2 - 2) if r2 > 2 else np.inf
        stats[key] = dict(n=n, ov=ov, s=s, seeds=seeds, mean_db=mu, mean_ci=(mlo, mhi), m2=r2,
                          m2_ci=(r2lo, r2hi), nu=nu, ptrust=pt, p10=p10)
        name = f"{arm:8s} U={U:4.1f} {surf:6s} {POLNAME[pol]} {a}°"
        P(f"{name:38s} {n:5d} {ov:4d} {mu:8.2f} [{mlo:6.2f},{mhi:6.2f}] {10*np.log10(bragg(pol,a)):7.2f} "
          f"{r2:6.2f} [{r2lo:5.2f},{r2hi:5.2f}] {nu:6.1f} {pt:8.1e}")
    # ---------------- paired choppy - gaussian ---------------------------------------------
    P("\nPAIRED choppy vs Gaussian (same seeds), 95% bootstrap CIs")
    P(f"{'case':28s} {'n':>5s} {'d<s0> dB':>22s} {'d m2':>22s} {'ratio P(>5<s>)':>24s} {'ratio P(>10<s>)':>24s}")
    paired = {}
    for key in keys:
        arm, U, surf, pol, a = key
        if surf != "gauss" or (arm, U, "choppy", pol, a) not in data:
            continue
        rg, rc = data[key], data[(arm, U, "choppy", pol, a)]
        common = sorted(q for q in set(rg) & set(rc) if not rg[q]["overturned"] and not rc[q]["overturned"])
        sg = np.array([rg[q]["sigma"] for q in common])
        sc = np.array([rc[q]["sigma"] for q in common])
        n = len(common)
        idx = rng.integers(0, n, (B, n))
        dmean = np.array([10 * np.log10(sc[i].mean() / sg[i].mean()) for i in idx])
        dm2 = np.array([m2(sc[i]) - m2(sg[i]) for i in idx])

        def pr(t, i):
            pg = np.mean(sg[i] / sg[i].mean() > t)
            pc = np.mean(sc[i] / sc[i].mean() > t)
            return pc / pg if pg > 0 else np.nan
        r5 = np.array([pr(5, i) for i in idx])
        r10 = np.array([pr(10, i) for i in idx])
        allidx = np.arange(n)
        point = (10 * np.log10(sc.mean() / sg.mean()), m2(sc) - m2(sg), pr(5, allidx), pr(10, allidx))
        cis = [ci(v) for v in (dmean, dm2, r5, r10)]
        paired[(U, pol, a)] = dict(point=point, ci=cis, n=n)
        name = f"U={U:4.1f} {POLNAME[pol]} {a}°"
        P(f"{name:28s} {n:5d} " + " ".join(f"{p:+7.2f} [{c[0]:+6.2f},{c[1]:+6.2f}]" for p, c in zip(point, cis)))
    # ---------------- diagnostics --------------------------------------------------------
    P("\nDIAGNOSTICS (per case: energy |ratio/pred - 1| median & max, MOMI it3 vs it2 max |dB|,"
      " edge |J| max, min dx/du, window h_rms, slope_rms, skewness)")
    for key in keys:
        recs = [r for r in data[key].values() if not r["overturned"]]
        e = np.array([r["energy"] / r["energy_pred"] - 1 for r in recs])
        it = np.array([abs(10 * np.log10(r["sigma"] / r["sigma_it2"])) for r in recs])
        ed = np.array([r["edge_J"] for r in recs])
        jm = np.array([r["jac_min_window"] for r in recs])
        jd = np.array([r["jac_min_domain"] for r in recs])
        arm, U, surf, pol, a = key
        sv = np.array([r["sigma"] for r in recs])
        flag = abs(e) > 0.02
        m2ok = m2(sv[~flag])
        P(f"{arm:8s} U={U:4.1f} {surf:6s} {POLNAME[pol]} {a}°: energy>2% in {flag.sum()}/{len(e)} (m2 without them {m2ok:.2f}) | "
          f"domain overturned in {np.sum(jd <= 0)}/{len(jd)} | energy {np.median(abs(e)):.1e}/{abs(e).max():.1e} | "
          f"MOMI {it.max():.1e} dB (99%: {np.percentile(it,99):.1e}) | edge {ed.max():.1e} | "
          f"min dx/du win {jm.min():.2f} (dom {jd.min():.2f}) | h_rms {np.mean([r['h_rms_win'] for r in recs]):.3f} m "
          f"(nominal {recs[0]['h_rms_nominal']:.3f}) | slope {np.mean([r['slope_rms_win'] for r in recs]):.3f} "
          f"(nominal {recs[0]['slope_rms_nominal']:.3f}) | skew {np.mean([r['skew_win'] for r in recs]):+.2f}")
    txt = "\n".join(lines)
    open(os.path.join(RES, f"summary{TAG}.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    if PATCH40:
        figures(stats, paired)


def band(ax, s, color, label, ls="-"):
    n = len(s)
    idx = rng.integers(0, n, (B, n))
    E = np.array([exceed(s[i]) for i in idx])
    lo, hi = ci(E)
    p = exceed(s)
    ok = p > 0
    ax.semilogy(TAU[ok], p[ok], color=color, ls=ls, label=f"{label} (n={n})")
    ax.fill_between(TAU, np.maximum(lo, 1e-6), np.maximum(hi, 1e-6), color=color, alpha=0.2, lw=0)


def figures(stats, paired):
    # 1) Gaussian vs choppy at U=10, both angles, both pols
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i, a in enumerate((60, 80)):
        for j, pol in enumerate(("TE", "TM")):
            ax = axs[i, j]
            for surf, c in (("gauss", "C0"), ("choppy", "C3")):
                key = ("full", 10.0, surf, pol, a)
                if key in stats:
                    band(ax, stats[key]["s"], c, f"{surf}, m2={stats[key]['m2']:.2f}")
            ax.semilogy(TAU, np.exp(-TAU), "k--", lw=1, label="exponential")
            key = ("full", 10.0, "gauss", pol, a)
            if key in stats:
                ax.axhline(stats[key]["ptrust"], color="gray", ls=":", lw=1, label="trust floor")
            ax.set(ylim=(1e-4, 1), xlim=(0, 16), title=f"U = 10 m/s, {POLNAME[pol]}, θ = {a}°",
                   xlabel="τ / ⟨σ⟩", ylabel="P(σ > τ)")
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_gauss_vs_choppy_U10.png"), dpi=130); plt.close(fig)

    # 2) wind sweep, 80 deg HH (and VV for context)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, surf in zip(axs, ("gauss", "choppy")):
        for U, c in ((5.0, "C2"), (10.0, "C0"), (15.0, "C3")):
            key = ("full", U, surf, "TE", 80)
            if key in stats:
                band(ax, stats[key]["s"], c, f"U={U:g}, m2={stats[key]['m2']:.2f}")
        if surf == "gauss" and ("periodic", 15.0, "gauss", "TE", 80) in stats:
            st = stats[("periodic", 15.0, "gauss", "TE", 80)]
            band(ax, st["s"], "k", f"U=15 periodic L_gen=L (old design), m2={st['m2']:.2f}", ls="-.")
        ax.semilogy(TAU, np.exp(-TAU), "k--", lw=1, label="exponential")
        ax.set(ylim=(1e-4, 1), xlim=(0, 20), title=f"HH, θ = 80°, {surf}", xlabel="τ / ⟨σ⟩", ylabel="P(σ > τ)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_wind_sweep_80HH.png"), dpi=130); plt.close(fig)

    # 3) mean levels vs Bragg, and m2 forest plot
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    items = [k for k in sorted(stats) if k[0] == "full"] + [k for k in sorted(stats) if k[0] == "periodic"]
    for y, key in enumerate(items):
        st = stats[key]
        arm, U, surf, pol, a = key
        c = "C0" if surf == "gauss" else "C3"
        d = st["mean_db"] - 10 * np.log10(bragg(pol, a))
        lo, hi = np.array(st["mean_ci"]) - 10 * np.log10(bragg(pol, a))
        axs[0].errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=c, mfc="w" if arm == "periodic" else c)
        axs[1].errorbar(st["m2"], y, xerr=[[st["m2"] - st["m2_ci"][0]], [st["m2_ci"][1] - st["m2"]]],
                        fmt="o", color=c, mfc="w" if arm == "periodic" else c)
    labels = [f"{'per ' if k[0]=='periodic' else ''}U{k[1]:g} {k[2]} {POLNAME[k[3]]} {k[4]}°" for k in items]
    for ax in axs:
        ax.set_yticks(range(len(items))); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
    axs[0].axvline(0, color="k", lw=1)
    axs[0].set(xlabel="⟨σ⁰⟩ − Bragg prediction [dB]", title="Mean level vs first-order Bragg")
    axs[1].axvline(2, color="k", lw=1)
    axs[1].set(xlabel="⟨σ²⟩/⟨σ⟩²  (2 = exponential)", title="Second-moment ratio (95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_means_and_m2.png"), dpi=130); plt.close(fig)

    # 4) paired differences
    if paired:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        ks = sorted(paired)
        for y, kk in enumerate(ks):
            pp = paired[kk]
            for ax, ii, ref in ((axs[0], 1, 0), (axs[1], 2, 1)):
                v, (lo, hi) = pp["point"][ii], pp["ci"][ii]
                ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color="C3")
                ax.axvline(ref, color="k", lw=1)
        for ax in axs:
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels([f"U{U:g} {POLNAME[p]} {a}°" for U, p, a in ks])
            ax.invert_yaxis()
        axs[0].set(xlabel="m2(choppy) − m2(Gaussian)", title="Paired difference in ⟨σ²⟩/⟨σ⟩² (95% CI)")
        axs[1].set(xlabel="P_choppy(σ>5⟨σ⟩) / P_gauss(σ>5⟨σ⟩)", title="Paired tail ratio at τ = 5⟨σ⟩ (95% CI)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_paired_differences.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
