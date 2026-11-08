"""TEST 5 - Random ocean-like surfaces with the Pierson-Moskowitz spectrum (thesis Ch. 3, 4.2).

What it teaches
  * Spectral method: each realization is a sum of sinusoids with random Gaussian amplitudes,
    E|c_m|^2 = W(K_m) dK. The ensemble-averaged periodogram must reproduce W(K).
  * P-M numbers from the thesis for U = 5 m/s: rms height 0.133 m (eq. 4.7), dominant
    wavelength Lambda_p = 22.8 m (eq. 4.4); with lambda = 23 cm that is k*h_rms = 3.6
    -- far too rough for perturbation theory, hence numerics.
  * W ~ K^-3 at high K, so the slope variance int K^2 W dK grows like log(Kc): the
    spectral cutoff Kc sets the slope (and curvature) statistics. This is why the
    choice of Kc (thesis uses 5k) matters so much, and why Toporkov recommends >= 3-4k for
    single-realization quantities like our tail probabilities.
  * The same realization can be re-sampled at any dx (we store the Fourier coefficients),
    which is how the sampling studies in Ch. 4 compare like with like.
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, savefig

lam, U, g0 = 0.23, 5.0, 9.81
k = 2 * np.pi / lam
L = 819.2 * lam                    # the thesis' surface length
Kc = 4.99 * k
W = lambda K: m.pm_spectrum(K, U)

Kp = np.sqrt(2 * 0.74 / 3) * g0 / U**2            # peak of K^-3 exp(-b/K^2)
check("dominant wavelength Lambda_p [m] (thesis 22.8)", 2 * np.pi / Kp, 22.8, 0.01)
check("rms height [m] (thesis 0.133)", m.pm_rms_height(U), 0.133, 0.01)
print(f"  k * h_rms = {k * m.pm_rms_height(U):.2f} (thesis: 3.6)")

rng = np.random.default_rng(0)
n_real, dx = 60, lam / 10
x = np.arange(int(L / dx)) * dx
dK = 2 * np.pi / L
Kgrid = dK * np.arange(1, int(Kc / dK) + 1)
var_h, var_s, pgram = [], [], 0
for _ in range(n_real):
    s = m.random_surface(L, W, Kc, rng)
    var_h.append(np.mean(s.f(x) ** 2))
    var_s.append(np.mean(s.fx(x) ** 2))
    pgram = pgram + np.abs(s.c) ** 2 / dK / n_real        # estimate of W(K_m)
print(f"averaged over {n_real} realizations, L = {L:.0f} m:")
check("rms height (ensemble) vs discrete-spectrum prediction",
      np.sqrt(np.mean(var_h)), np.sqrt(2 * np.sum(W(Kgrid)) * dK), 0.04)
slope_pred = 2 * np.sum(Kgrid**2 * W(Kgrid)) * dK
check("rms slope vs int K^2 W dK", np.sqrt(np.mean(var_s)), np.sqrt(slope_pred), 0.03)
check("periodogram / W(K), mean over K", np.mean(pgram / W(Kgrid)), 1.0, 0.05)

s = m.random_surface(L, W, Kc, np.random.default_rng(1))
xa, za = s.sample(lam / 10)[:2]
xb, zb = s.sample(lam / 40)[:2]
check("resampling: same surface at dx and dx/4 (max diff, m)", np.max(abs(za - zb[::4])), 0, 1e-12, rel=False)

print("\nrms slope vs spectral cutoff (log growth -> the cutoff controls slope statistics):")
for c in [1, 2, 3, 5, 10]:
    KK = dK * np.arange(1, int(c * k / dK) + 1)
    print(f"  Kc = {c:2d} k : rms slope = {np.degrees(np.arctan(np.sqrt(2*np.sum(KK**2*W(KK))*dK))):.2f} deg")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(xa, za, lw=0.6)
axs[0].set(xlabel="x [m]", ylabel="ζ [m]", title=f"P-M realization, U={U} m/s, L={L:.0f} m")
axs[1].loglog(Kgrid, pgram, ".", ms=2, alpha=0.5, label=f"periodogram, {n_real} realizations")
axs[1].loglog(Kgrid, W(Kgrid), "k", label="W(K), eq. 4.2")
for th in [20, 50, 80]:
    axs[1].axvline(2 * k * np.sin(np.radians(th)), ls=":", c="C3")
axs[1].text(2 * k * np.sin(np.radians(80)), W(Kgrid).max() * 1e-3, " Bragg K=2k sinθ\n (20°,50°,80°)", fontsize=8)
axs[1].set(xlabel="K [rad/m]", ylabel="W(K) [m³]", title="Spectrum check")
axs[1].legend()
savefig(fig, "05_pm_surfaces.png")
summary()
