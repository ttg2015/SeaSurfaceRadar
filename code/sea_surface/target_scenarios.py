"""Best- and worst-case pictures of a person in the water, X-band-resolved surfaces.

2D model: the person is an infinite cylinder out of the plane, so we draw a 1D cut.
Target = elliptical cap rising above the local sea surface, h_total = max(h_sea, cap).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sea_surface_1d import surface
from sea_surface_xband import elfouhaily_k, coeffs

lam_em = 0.03
k_cut = 2.5 * 2 * np.pi / lam_em
L, N = 20.0, 2**14  # 1.2 mm grid

scenarios = {
    "Best case: calm sea (U10 = 3 m/s), torso + life vest on a crest": dict(
        U10=3.0, width=0.5, height=0.25, where="crest", seed=3),
    "Worst case: rough sea (U10 = 12 m/s), head only in a trough": dict(
        U10=12.0, width=0.18, height=0.12, where="trough", seed=4),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw=dict(width_ratios=[3, 1]))
x = np.arange(N) * L / N
for (title, s), (ax, axz) in zip(scenarios.items(), axes):
    k, c = coeffs(N, L, lambda kk: elfouhaily_k(kk, s["U10"]), k_cut, np.random.default_rng(s["seed"]))
    _, h = surface(k, c)
    h -= h.mean()
    mid = slice(N // 4, 3 * N // 4)
    x0 = x[mid][np.argmax(h[mid]) if s["where"] == "crest" else np.argmin(h[mid])]
    base = h[np.argmin(np.abs(x - x0))]
    a, b = s["width"] / 2, s["height"]
    cap = base + b * np.sqrt(np.clip(1 - ((x - x0) / a) ** 2, 0, None))
    cap[np.abs(x - x0) > a] = -np.inf
    ht = np.maximum(h, cap)

    for a_ in (ax, axz):
        a_.fill_between(x, ht, ht.min() - 0.3, color="#9cc3e6")
        a_.plot(x, ht, c="#1f4e79", lw=0.8)
        a_.fill_between(x, h, ht, where=ht > h, color="#d95f02", label="person")
        a_.set_xlabel("x [m]"); a_.set_ylabel("z [m]")
    ax.set_title(title, fontsize=10)
    ax.annotate("", xy=(x0, base + b + 0.05), xytext=(x0 - 3, base + b + 0.6),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_aspect("equal")
    axz.set_xlim(x0 - 1, x0 + 1); axz.set_aspect("equal")
    axz.set_ylim(base - 0.5, base + 0.5)
    axz.set_title("2 m zoom", fontsize=10)
    print(f"{title}: Hs = {4*h.std():.2f} m, exposed height {b} m, width {s['width']} m")

fig.tight_layout()
fig.savefig("target_scenarios.png", dpi=130)
