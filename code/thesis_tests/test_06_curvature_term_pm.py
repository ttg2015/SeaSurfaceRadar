"""TEST 6 - The curvature term on ocean-like surfaces (thesis Ch. 4, the main numerical finding).

What it teaches
  * Same P-M realization, sampled at dx = lambda/10, /20, /40. Without the curvature
    term on the diagonal, the backscattered NRCS of a single realization drifts with dx;
    worst for horizontal (TE / HH) polarization near grazing, exactly as Toporkov found.
  * With the curvature term the answer is already stable at lambda/10.
  * Why it matters for the project: our deliverable is a tail probability of
    single-realization values, so a dx-dependent bias of ~1 dB per realization would shift
    the estimated false-alarm rate directly.
  Note: the 80 deg case uses g = L/6 = 17 lambda, below the Kapp criterion (~49 lambda), so
  absolute values there are not trustworthy -- but the dx sensitivity is still visible.
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, savefig

lam, U = 0.23, 5.0
k = 2 * np.pi / lam
L = 102.4 * lam
g = L / 6
ns = [10, 20, 40]
angles = [50, 70, 80]
n_real = 3

res = {}  # (pol, curv, real, theta) -> [dB at each n]
for r in range(n_real):
    surf = m.random_surface(L, lambda K: m.pm_spectrum(K, U), 4.99 * k, np.random.default_rng(100 + r))
    for n in ns:
        x, z, zx, zxx = surf.sample(lam / n)
        for pol in ["TE", "TM"]:
            P = m.propagator(x, z, zx, zxx, k, pol, curvature=True)
            diag = np.diag(P).copy()
            for curv in [True, False]:
                np.fill_diagonal(P, diag if curv else 0.0)
                for thd in angles:
                    th = np.radians(thd)
                    J = m.solve_direct(P, m.kirchhoff_current(x, z, zx, k, th, g, pol))
                    s = m.nrcs(x, z, zx, J, k, th, g, -th, pol)[0]
                    res.setdefault((pol, curv, r, thd), []).append(10 * np.log10(s))
    print(f"realization {r} done")

print("\nBackscatter NRCS [dB] at dx = lambda/10, /20, /40   (spread = max - min)")
spread = {}
for pol in ["TE", "TM"]:
    for curv in [False, True]:
        for thd in angles:
            sp = [np.ptp(res[(pol, curv, r, thd)]) for r in range(n_real)]
            spread[(pol, curv, thd)] = max(sp)
            vals = "  ".join("[" + " ".join(f"{v:6.2f}" for v in res[(pol, curv, r, thd)]) + "]"
                             for r in range(n_real))
            print(f"  {pol} {'curv ' if curv else 'nocurv'} {thd}°: {vals}  max spread {max(sp):.2f} dB")

check("TE 80 deg, no curvature: dx-sensitivity > 0.5 dB", float(spread[("TE", False, 80)] > 0.5), 1, 0)
check("TE 80 deg, with curvature: dx-sensitivity < 0.3 dB", spread[("TE", True, 80)], 0, 0.3, rel=False)
check("curvature term reduces TE 80 deg sensitivity > 3x",
      float(spread[("TE", False, 80)] > 3 * spread[("TE", True, 80)]), 1, 0)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pol in zip(axs, ["TE", "TM"]):
    for thd, c in zip(angles, ["C0", "C1", "C3"]):
        for curv, ls in [(False, "--"), (True, "-")]:
            y = [spread[(pol, curv, thd)]]
            ax.bar(angles.index(thd) * 3 + (0 if not curv else 1), y, color=c,
                   alpha=0.4 if not curv else 1.0, hatch="//" if not curv else None)
    ax.set_xticks([0.5, 3.5, 6.5], [f"{a}°" for a in angles])
    ax.set(title=f"{pol} ({'HH' if pol == 'TE' else 'VV'}): hatched = no curvature term",
           xlabel="incidence angle")
axs[0].set_ylabel("max spread of σ_bsc over Δx = λ/10..λ/40 [dB]")
savefig(fig, "06_curvature_term_pm.png")
summary()
