"""Long-form video of 1D choppy waves: Elfouhaily spectrum, gravity-capillary dispersion,
Lagrangian (Tessendorf/Creamer-style) horizontal displacement.

Top: full 400 m periodic domain (vertical exaggeration). Bottom: 6 m window (camera follows local mean height) at true scale.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from sea_surface_1d import surface
from sea_surface_xband import elfouhaily_k, coeffs

U10 = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
T_total = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
fps = 30                           # video time = simulated time
L, N = 400.0, 2**17                # 3 mm grid
k_cut = 2.5 * 2 * np.pi / 0.03     # resolves X-band Bragg waves
rng = np.random.default_rng(7)
k, c = coeffs(N, L, lambda kk: elfouhaily_k(kk, U10), k_cut, rng)
x = np.arange(N) * L / N

D0, h0 = surface(k, c, lam=1.0)
Hs = 4 * h0.std()
zx0, zw = 200.0, 6.0               # zoom window

sky, sea, deep, foam = "#0b1d2e", "#2a6f97", "#012a4a", "#e8f1f8"
plt.rcParams.update({"text.color": "w", "axes.labelcolor": "w", "xtick.color": "w",
                     "ytick.color": "w", "axes.edgecolor": "#5c7a94"})
fig, (ax, az) = plt.subplots(2, 1, figsize=(16, 9), facecolor=sky,
                             gridspec_kw=dict(height_ratios=[1, 1.3]))
for a in (ax, az):
    a.set_facecolor(sky)
ylim = 1.6 * Hs
ax.set_xlim(0, L); ax.set_ylim(-ylim, ylim)
ax.set_ylabel("z [m]"); ax.set_xlabel("x [m]")
az.set_xlim(zx0, zx0 + zw); az.set_aspect("equal", adjustable="datalim")
az.set_ylabel("z [m]"); az.set_xlabel("x [m]  (true scale)")
ttl = fig.suptitle("", color="w", fontsize=14)
ax.axvspan(zx0, zx0 + zw, color="w", alpha=0.08)

state = {}


def draw(i):
    t = i / fps
    D, h = surface(k, c, t, lam=1.0)
    X = x + D
    order = np.argsort(X)          # periodic wrap at domain edges
    Xs, hs = np.mod(X[order], L), h[order]
    o2 = np.argsort(Xs); Xs, hs = Xs[o2], hs[o2]
    slope = np.gradient(hs, Xs + 1e-9 * np.arange(N))
    for kk in ("f1", "f2", "l1", "l2", "s1", "s2"):
        if kk in state:
            state[kk].remove()
    sub = slice(None, None, 8)
    state["f1"] = ax.fill_between(Xs[sub], hs[sub], -ylim, color=sea, lw=0)
    state["l1"], = ax.plot(Xs[sub], hs[sub], c=foam, lw=0.6)
    m = (Xs > zx0 - 1) & (Xs < zx0 + zw + 1)
    state["f2"] = az.fill_between(Xs[m], hs[m], -zw, color=deep, lw=0)
    state["l2"], = az.plot(Xs[m], hs[m], c=foam, lw=1.0)
    zc = hs[m].mean()
    az.set_ylim(zc - 0.6, zc + 0.6)
    steep = m & (np.abs(slope) > 0.6)   # crude whitecap marker
    state["s1"] = az.scatter(Xs[steep], hs[steep], s=4, c="w", alpha=0.7, lw=0)
    state["s2"] = ax.scatter(Xs[steep], hs[steep], s=2, c="w", alpha=0.5, lw=0)
    ttl.set_text(f"1D choppy sea   U10 = {U10:.0f} m/s   Hs = {Hs:.2f} m   "
                 f"dx = {1000*L/N:.0f} mm   t = {t:6.1f} s")
    return []


fig.tight_layout(rect=(0, 0, 1, 0.96))
out = f"choppy_waves_U{U10:.0f}.mp4"
FuncAnimation(fig, draw, frames=int(T_total * fps)).save(
    out, writer=FFMpegWriter(fps=fps, bitrate=8000), dpi=100, savefig_kwargs={"facecolor": sky})
print("wrote", out)
