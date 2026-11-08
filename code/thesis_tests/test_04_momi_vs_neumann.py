"""TEST 4 - Iterative solvers: Neumann series vs MOMI (thesis Sec. 2.2).

What it teaches
  * The plain Neumann series J = J_i + P J_i + P^2 J_i + ... adds one more bounce per
    term. It converges only if the spectral radius of P is < 1 (not guaranteed; it can
    diverge for strong multiple scattering [Wingham & Devayya 1992]), and even when it
    converges it is slow at low grazing, because energy keeps hopping along the surface.
  * MOMI splits P = L + U (left / right of the observer). (I-L)^-1 is ONE forward sweep that
    sums ALL orders of forward multiple scattering exactly; (I-U)^-1 does the same backward.
    The remaining coupling L U (forward then backward) is weak, so:
        new Born term alone      -> already close   (thesis: "often gives a reasonable answer")
        + 2 iterations           -> converged       (thesis: "more than adequate")
    Triangular solves mean no matrix inverse and no stored matrix (here we store it for clarity).
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, savefig

lam = 0.23                              # thesis wavelength (m), Ch. 4
k = 2 * np.pi / lam
U = 5.0
L = 102.4 * lam
dx = lam / 10
surf = m.random_surface(L, lambda K: m.pm_spectrum(K, U), 4.99 * k, np.random.default_rng(7))
x, z, zx, zxx = surf.sample(dx)
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pol in zip(axs, ["TE", "TM"]):
    P = m.propagator(x, z, zx, zxx, k, pol)
    rho = np.max(np.abs(np.linalg.eigvals(P)))
    print(f"{pol}: spectral radius of P = {rho:.2f}  (Neumann series converges only if < 1)")
    for thd, c in [(20, "C0"), (50, "C1"), (80, "C3")]:
        th = np.radians(thd)
        Ji = m.kirchhoff_current(x, z, zx, k, th, L / 6, pol)
        Jd = m.solve_direct(P, Ji)
        e_m = [rel(J, Jd) for J in m.solve_momi(P, Ji, 6)]
        e_n = [rel(J, Jd) for J in m.solve_neumann(P, Ji, 6)]
        ax.semilogy(range(7), e_m, "o-", c=c, label=f"MOMI θi={thd}°")
        ax.semilogy(range(1, 7), e_n, "x--", c=c, label=f"Neumann θi={thd}°")
        print(f"  theta_i={thd}: MOMI  err " + " ".join(f"{e:.1e}" for e in e_m))
        print(f"             Neumann err " + " ".join(f"{e:.1e}" for e in e_n))
        check(f"  MOMI after 2 iterations ({pol}, {thd} deg) < 1e-2", e_m[2], 0, 1e-2, rel=False)
        check(f"  MOMI beats Neumann at equal cost ({pol}, {thd} deg)", float(e_m[2] < e_n[2]), 1, 0)
    ax.set(xlabel="iteration (0 = new Born term for MOMI)", title=f"{pol}: P-M surface, U=5 m/s")
    ax.legend(fontsize=7)
axs[0].set_ylabel("rel. error vs direct solve")
savefig(fig, "04_momi_vs_neumann.png")
summary()
