"""TEST 7 - Mini Monte Carlo: the project's deliverable P(sigma > tau), in miniature.

What it teaches
  * Each random surface gives ONE backscatter number sigma_bsc (|A(theta)|^2 up to a constant).
    Averaging gives the mean NRCS <sigma0> that the thesis reports (eq. 2.34 + ensemble avg).
    The *distribution* of single-realization values is what a false-alarm rate needs.
  * If the illuminated patch contains many independent scatterers, the far field is a sum of
    many random phasors -> complex Gaussian -> sigma is EXPONENTIAL ("speckle"):
        P(sigma > tau) = exp(-tau / <sigma>),   <sigma^2>/<sigma>^2 = 2.
    Sea spikes = heavier-than-exponential tails. Gaussian linear P-M surfaces with a small
    patch should be close to exponential; deviations are the interesting physics.
  * Mean level vs first-order small-perturbation (Bragg) theory, derived in test 2:
        HH: <sigma0> = 8 pi k^3 cos^4(theta) W(2k sin theta)
        VV: <sigma0> = 8 pi k^3 (1 + sin^2 theta)^2 W(2k sin theta)
    The P-M surface is NOT small (k h_rms = 3.6), but the large waves mostly tilt the Bragg
    ripples ("two-scale" picture), so VV is usually near SPM while HH departs more.
  Parameters are deliberately small (L ~ 24 m, 300 realizations) so this runs in ~1-2 min.
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, savefig

lam, U = 0.23, 5.0
k = 2 * np.pi / lam
L = 102.4 * lam
dx = lam / 10
g = L / 6
th = np.radians(60)
n_real = 300
W = lambda K: m.pm_spectrum(K, U)

rng = np.random.default_rng(2026)
sig = {"TE": [], "TM": []}
for r in range(n_real):
    surf = m.random_surface(L, W, 4.99 * k, rng)
    x, z, zx, zxx = surf.sample(dx)
    for pol in sig:
        P = m.propagator(x, z, zx, zxx, k, pol)
        J = m.solve_momi(P, m.kirchhoff_current(x, z, zx, k, th, g, pol), 2)[-1]
        sig[pol].append(m.nrcs(x, z, zx, J, k, th, g, -th, pol)[0])
    if (r + 1) % 50 == 0:
        print(f"  {r+1}/{n_real} realizations")

Wb = W(2 * k * np.sin(th))
spm = {"TE": 8 * np.pi * k**3 * np.cos(th) ** 4 * Wb,
       "TM": 8 * np.pi * k**3 * (1 + np.sin(th) ** 2) ** 2 * Wb}
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for pol, c in [("TE", "C0"), ("TM", "C3")]:
    s = np.array(sig[pol])
    mean = s.mean()
    se = s.std() / np.sqrt(n_real)
    name = "HH" if pol == "TE" else "VV"
    print(f"\n{name} (theta_i = 60 deg): <sigma0> = {10*np.log10(mean):.2f} dB "
          f"(+/- {10*np.log10(1+se/mean):.2f}),  SPM = {10*np.log10(spm[pol]):.2f} dB")
    m2 = np.mean(s**2) / mean**2
    print(f"  <s^2>/<s>^2 = {m2:.2f}  (exponential: 2)")
    check(f"{name}: second moment ratio near exponential's 2", m2, 2.0, 0.3)
    if pol == "TM":
        check("VV mean within 3 dB of SPM", 10 * np.log10(mean / spm[pol]), 0, 3, rel=False)
    tau = np.sort(s / mean)
    exceed = 1 - np.arange(n_real) / n_real
    axs[0].semilogy(tau, exceed, c, label=f"{name} Monte Carlo")
    axs[1].hist(10 * np.log10(s), bins=30, alpha=0.5, color=c, label=name)
    axs[1].axvline(10 * np.log10(spm[pol]), color=c, ls="--")
t = np.linspace(0, 6, 100)
axs[0].semilogy(t, np.exp(-t), "k--", label="exponential exp(−τ)")
axs[0].set(xlabel="τ / ⟨σ⟩", ylabel="P(σ_bsc > τ)", title="Exceedance probability (the deliverable)")
axs[0].legend()
axs[1].set(xlabel="σ_bsc [dB]", ylabel="count", title="Single-realization backscatter (dashed: SPM mean)")
axs[1].legend()
savefig(fig, "07_monte_carlo_tail.png")
summary()
