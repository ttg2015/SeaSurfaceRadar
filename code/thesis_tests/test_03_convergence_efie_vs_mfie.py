"""TEST 3 - Discretization, the curvature term, and why the incident field must solve Helmholtz.

What it teaches (three separate lessons, all from Ch. 4-5 of the thesis)

  A. The MFIE kernel is bounded, but its diagonal LIMIT is the curvature term
     zeta'' dx / (2 pi (1 + zeta'^2))   (eq. 4.1), not zero. Applying the discrete operator
     to a smooth, compactly supported function:
        with curvature term  -> error ~ dx^3   (the only non-smoothness is d^2 log|d|)
        zero diagonal        -> error ~ dx     (an O(dx) piece of every row is missing)

  B. In the full TE solve the current does NOT die off where the beam does: energy is
     carried along the surface and J at the downstream truncation edge stays at a few % of
     its peak. Cutting the surface there makes the whole solve first order in dx again,
     whatever the diagonal. The curvature term still cuts the error ~7x. This is the
     "slowly decaying tail in the surface current" for TE that Toporkov links to taper
     failure; the fix is a longer surface and a wider beam (Kapp criterion, eq. 5.8).

  C. EFIE (2.10, uses E_i) and MFIE (2.12, uses dE_i/dn) are the same problem only if the
     incident field exactly satisfies Helmholtz. The Thorsos taper does not (the residual
     grows like 1/(k g cos theta)^2): for a narrow beam the two equations converge to answers
     ~20% apart. With an exact beam (a sum of plane waves) they agree. This is why the thesis
     spends a whole chapter (Ch. 5) on tapers.
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, savefig

k = 2 * np.pi
L = 16.0
th = np.radians(60)
g = L / 7
surf = m.sinusoid_surface(L, 0.15, 1.5)     # max slope 0.63, strongly curved
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# ---- A: operator quadrature -------------------------------------------------------------
print("A. Discrete operator applied to a smooth test function, error vs dx")
f = lambda x: np.exp(-(x / 2) ** 2) * np.exp(3j * x)
ns = np.array([10, 20, 40, 80, 160])
order = {}
for curv, lab in [(True, "curvature term"), (False, "zero diagonal")]:
    v = []
    for n in list(ns) + [640]:
        x, z, zx, zxx = surf.sample(1 / n)
        i = np.argmin(abs(x - 0.3))
        v.append(m.propagator(x, z, zx, zxx, k, "TE", curv)[i] @ f(x))
    err = np.abs(np.array(v[:-1]) - v[-1])
    order[lab] = np.polyfit(np.log(1 / ns), np.log(err), 1)[0]
    axs[0].loglog(ns, err, "o-", label=f"{lab} (order {order[lab]:.1f})")
    print(f"  {lab}: errors {' '.join(f'{e:.1e}' for e in err)}  order {order[lab]:.2f}")
check("operator order with curvature term", order["curvature term"], 3, 0.4, rel=False)
check("operator order with zero diagonal", order["zero diagonal"], 1, 0.3, rel=False)
axs[0].set(xlabel="λ/Δx", ylabel="|error| of (P f)(x0)", title="A. Quadrature of the MFIE operator")
axs[0].legend()

# ---- B: full solve -------------------------------------------------------------------------
print("\nB. Full TE solve, successive-refinement differences of J")
diffs = {}
for curv, lab in [(True, "curvature term"), (False, "zero diagonal")]:
    S = {}
    for n in [10, 20, 40, 80]:
        x, z, zx, zxx = surf.sample(1 / n)
        J = m.solve_direct(m.propagator(x, z, zx, zxx, k, "TE", curv),
                           m.kirchhoff_current(x, z, zx, k, th, g, "TE"))
        S[n] = J[:: n // 10]
    diffs[lab] = [rel(S[n], S[2 * n]) for n in [10, 20, 40]]
    print(f"  {lab}: {' '.join(f'{e:.1e}' for e in diffs[lab])}")
    if curv:
        Jfine, xfine, Ji = J, x, m.kirchhoff_current(x, z, zx, k, th, g, "TE")
check("curvature term reduces error at dx=lambda/10 by > 4x",
      diffs["zero diagonal"][0] / diffs["curvature term"][0], 7, 0.45)
edge = abs(Jfine[-1]) / abs(Jfine).max()
print(f"  |J| at downstream edge / max = {edge:.1e}  (forcing there: {abs(Ji[-1])/abs(Ji).max():.0e})")
axs[1].semilogy(xfine, abs(Ji), "k--", label="Kirchhoff current |J_i|")
axs[1].semilogy(xfine, abs(Jfine), "C0", label="MFIE current |J|")
axs[1].set(xlabel="x [λ]", ylim=(1e-6, None), title="B. TE current outlives the beam")
axs[1].legend()

# ---- C: EFIE vs MFIE with Thorsos taper vs exact beam ---------------------------------------
print("\nC. EFIE vs MFIE (dx = lambda/80)")
x, z, zx, zxx = surf.sample(1 / 80)
P = m.propagator(x, z, zx, zxx, k, "TE")
A = m.efie_matrix_TE(x, z, zx, k)
gap = {}
for name, fld in [("Thorsos taper", m.thorsos_field), ("exact beam", m.exact_beam_field)]:
    Jm = m.solve_direct(P, m.kirchhoff_current(x, z, zx, k, th, g, "TE", fld))
    Je = np.linalg.solve(A, fld(x, z, k, th, g)[0])
    gap[name] = rel(Je, Jm)
    print(f"  {name:14s}: ||J_EFIE - J_MFIE|| / ||J_MFIE|| = {gap[name]:.3f}")
    axs[2].plot(x, abs(Jm), label=f"MFIE, {name}")
    axs[2].plot(x, abs(Je), "--", label=f"EFIE, {name}")
check("exact beam: EFIE ~ MFIE (< 5%)", gap["exact beam"], 0, 0.05, rel=False)
check("Thorsos taper gap is much larger than exact-beam gap",
      gap["Thorsos taper"] / gap["exact beam"], 9, 0.6)
axs[2].set(xlabel="x [λ]", ylabel="|J|", xlim=(-6, 6),
           title=f"C. EFIE vs MFIE, k g cosθ = {k*g*np.cos(th):.1f}")
axs[2].legend(fontsize=7)
savefig(fig, "03_convergence.png")
summary()
