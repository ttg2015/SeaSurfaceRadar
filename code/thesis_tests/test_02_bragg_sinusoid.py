"""TEST 2 - Sinusoidal surface: Bragg scattering, the mechanism behind sea clutter.

What it teaches
  * A periodic surface zeta = h cos(Kx) scatters only into discrete diffraction orders
        k sin(theta_n) = k sin(theta_i) + n K          (grating equation)
    With our angle convention, backscatter is theta_s = -theta_i, i.e. order n = -1 with
        K = 2 k sin(theta_i)     <=>   Lambda = lambda / (2 sin theta_i)   ("Bragg resonance").
    A random sea is a superposition of sinusoids; at low grazing the backscatter is dominated
    by the Bragg component of the spectrum, W(2k sin theta_i). This is why the thesis cares so
    much about the high-K tail and spectral cutoff Kc of the P-M spectrum (Sec. 4.4).
  * For small kh the power in each order is given by first-order perturbation theory (SPM),
    derived by expanding the BC about z=0 (see README):
        TE (Dirichlet, HH):  eta_n = (kh)^2 cos(theta_i) cos(theta_n)
        TM (Neumann,  VV):   eta_n = h^2 (k^2 - k sin(theta_i) k sin(theta_n))^2 / (k^2 cos theta_i cos theta_n)
    For backscatter the VV/HH ratio is ((1 + sin^2 theta_i) / cos^2 theta_i)^2 -> huge near
    grazing: the classic "VV > HH at low grazing" for a perfectly conducting sea.
  * Tapered beam -> each order becomes a lobe; the *area* of the lobe is what to compare:
        int_lobe sigma0 d theta_s  =  2 pi cos(theta_i) * eta_n
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, energy_ratio, savefig

k = 2 * np.pi
L, dx = 60.0, 0.05
ts = np.radians(np.linspace(-89.95, 89.95, 7197))


def lobe_area(sig, center, half_width_deg=12):
    w = np.abs(ts - center) < np.radians(half_width_deg)
    return np.trapezoid(sig[w], ts[w])


def spm_eff(pol, th_i, th_n, h):
    if pol == "TE":
        return (k * h) ** 2 * np.cos(th_i) * np.cos(th_n)
    return h**2 * (k**2 - k**2 * np.sin(th_i) * np.sin(th_n)) ** 2 / (k**2 * np.cos(th_i) * np.cos(th_n))


# ---- Part A: one Bragg-resonant grating, look at the whole bistatic pattern -------------
th_i = np.radians(30)
Lam = 1 / (2 * np.sin(th_i))       # = lambda: only orders n=0 (specular) and n=-1 (backscatter) propagate
g = L / 6
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for pol, c in [("TE", "C0"), ("TM", "C3")]:
    for h in [0.01, 0.05]:
        print(f"Part A: theta_i=30, Lambda={Lam:.2f} lambda, kh={k*h:.3f}, {pol}")
        sig, d = m.scatter(m.sinusoid_surface(L, h, Lam), dx, k, th_i, g, pol, ts)
        pred = 1 - (1 + 2 * np.tan(th_i) ** 2) / (2 * (k * g * np.cos(th_i)) ** 2)
        check("energy ratio", energy_ratio(sig, ts, th_i), pred, 2e-3)
        eta = lobe_area(sig, -th_i) / (2 * np.pi * np.cos(th_i))
        check("Bragg (n=-1) efficiency vs SPM", eta, spm_eff(pol, th_i, -th_i, h),
              0.03 if h == 0.01 else 0.5)
        axs[0].semilogy(np.degrees(ts), sig, c, ls="-" if h == 0.01 else ":",
                        label=f"{pol}, kh={k*h:.2f}")
axs[0].set(xlabel="θs [deg]", ylabel="σ0", ylim=(1e-7, None),
           title="Bragg-resonant sinusoid, θi=30°: only specular (+30°) and backscatter (−30°)")
axs[0].legend(fontsize=8)

# ---- Part B: sweep incidence angle, always at Bragg resonance -> HH and VV backscatter ---
h = 0.005
angles = np.arange(10, 81, 10)
eff = {"TE": [], "TM": []}
for thd in angles:
    th = np.radians(thd)
    g = max(L / 6, 1.2 * m.kapp_min_waist(k, th))
    Lb = max(L, 6 * g)
    for pol in eff:
        sig, _ = m.scatter(m.sinusoid_surface(Lb, h, 1 / (2 * np.sin(th))), dx, k, th, g, pol, ts)
        eff[pol].append(lobe_area(sig, -th, 6) / (2 * np.pi * np.cos(th)))
print(f"\nPart B: Bragg backscatter efficiency vs angle (kh={k*h:.3f})")
for i, thd in enumerate(angles):
    th = np.radians(thd)
    print(f"  {thd:2d} deg  HH {eff['TE'][i]:.3e} (SPM {spm_eff('TE', th, -th, h):.3e})   "
          f"VV {eff['TM'][i]:.3e} (SPM {spm_eff('TM', th, -th, h):.3e})")
    check(f"  VV/HH ratio at {thd} deg", eff["TM"][i] / eff["TE"][i],
          ((1 + np.sin(th) ** 2) / np.cos(th) ** 2) ** 2, 0.05)
tt = np.radians(np.linspace(5, 85, 200))
axs[1].semilogy(np.degrees(tt), spm_eff("TE", tt, -tt, h), "C0", label="HH SPM")
axs[1].semilogy(np.degrees(tt), spm_eff("TM", tt, -tt, h), "C3", label="VV SPM")
axs[1].semilogy(angles, eff["TE"], "C0o", label="HH MFIE")
axs[1].semilogy(angles, eff["TM"], "C3s", label="VV MFIE")
axs[1].set(xlabel="θi [deg]", ylabel="Bragg backscatter efficiency",
           title=f"Resonant Bragg backscatter, kh={k*h:.3f}")
axs[1].legend()
savefig(fig, "02_bragg_sinusoid.png")
summary()
