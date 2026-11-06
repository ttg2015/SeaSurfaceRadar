"""1D random sea surface resolved for X-band scattering.

Spectrum: Elfouhaily et al. (1997) omnidirectional, which includes the short-gravity /
capillary waves that cause Bragg scattering. JONSWAP has no such tail.
Grid spacing: lambda_em / 10. The spectrum is cut off at k_cut so that the
shortest surface wave still gets several grid points.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sea_surface_1d import g, omega, surface

c_light = 2.998e8
km, cm = 370.0, 0.23  # gravity-capillary minimum: wavenumber [rad/m], phase speed [m/s]


def elfouhaily_k(k, U10=10.0, Omega=0.84):
    """Elfouhaily omnidirectional spectrum S(k) [m^3]; U10 wind [m/s], Omega inverse wave age."""
    k = np.asarray(k, dtype=float)
    S = np.zeros_like(k)
    m = k > 0
    k = k[m]
    kp = g * Omega**2 / U10**2
    c = np.sqrt(g / k * (1 + (k / km) ** 2))
    cp = np.sqrt(g / kp * (1 + (kp / km) ** 2))

    gamma = 1.7 if Omega < 1 else 1.7 + 6 * np.log10(Omega)
    sig = 0.08 * (1 + 4 * Omega**-3)
    Gam = np.exp(-((np.sqrt(k / kp) - 1) ** 2) / (2 * sig**2))
    L_PM = np.exp(-1.25 * (kp / k) ** 2)
    Jp = gamma**Gam

    # long-wave curvature spectrum
    alpha_p = 6e-3 * np.sqrt(Omega)
    Fp = L_PM * Jp * np.exp(-Omega / np.sqrt(10) * (np.sqrt(k / kp) - 1))
    Bl = 0.5 * alpha_p * (cp / c) * Fp

    # short-wave curvature spectrum
    ustar = np.sqrt((0.8 + 0.065 * U10) * 1e-3) * U10
    r = np.log(ustar / cm)
    alpha_m = 0.01 * (1 + (r if ustar < cm else 3 * r))
    Fm = L_PM * Jp * np.exp(-0.25 * (k / km - 1) ** 2)
    Bh = 0.5 * alpha_m * (cm / c) * Fm

    S[m] = (Bl + Bh) / k**3
    return S


def coeffs(N, L, S_func, k_cut, rng):
    """Fourier coefficients for h = 2 Re sum_{k>0} c_k e^{ikx}; modes above k_cut zeroed."""
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    dk = 2 * np.pi / L
    S = np.where((k > 0) & (k <= k_cut), S_func(np.abs(k)), 0.0)
    amp = np.sqrt(S * dk / 4)
    return k, amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N))


if __name__ == "__main__":
    f_em = 10e9                       # X-band
    lam_em = c_light / f_em           # 3.0 cm
    k_em = 2 * np.pi / lam_em
    dx = lam_em / 10
    L = 200.0                         # [m]; should be several peak wavelengths for stats
    N = int(2 ** np.ceil(np.log2(L / dx)))
    dx = L / N
    k_cut = 2.5 * k_em                # > Bragg k at any angle (2 k_em), ~6 pts per shortest wave
    U10 = 10.0

    kp = g * 0.84**2 / U10**2
    print(f"lambda_em = {100*lam_em:.1f} cm, dx = {1000*dx:.2f} mm, N = {N}")
    print(f"peak wavelength = {2*np.pi/kp:.0f} m, domain = {L:.0f} m")
    print(f"k_cut = {k_cut:.0f} rad/m (shortest wave {100*2*np.pi/k_cut:.1f} cm,"
          f" {2*np.pi/k_cut/dx:.1f} pts)")

    rng = np.random.default_rng(1)
    k, c = coeffs(N, L, lambda kk: elfouhaily_k(kk, U10), k_cut, rng)
    x = np.arange(N) * dx
    _, h = surface(k, c)
    Dx, hc = surface(k, c, lam=1.0)

    kk = np.linspace(2 * np.pi / L, k_cut, 200000)
    Sk = elfouhaily_k(kk, U10)
    mss_model = np.trapezoid(kk**2 * Sk, kk)
    print(f"Hs = {4*h.std():.2f} m (spectrum {4*np.sqrt(np.trapezoid(Sk, kk)):.2f} m)")
    print(f"slope var = {np.var(np.gradient(h, dx)):.4f} (spectrum {mss_model:.4f}, "
          f"Cox-Munk total {3e-3 + 5.12e-3*U10:.4f})")
    fold = np.gradient(x + Dx, dx).min()
    print(f"choppy: min dX/dx = {fold:.2f} ({'FOLDS' if fold <= 0 else 'single-valued'})")
    print(f"memory for dense {N}x{N} complex matrix: {16*N**2/1e9:.0f} GB")
    np.savez("surface_xband.npz", x=x, h=h, x_choppy=x + Dx, h_choppy=hc, dx=dx, U10=U10, f_em=f_em)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(x, h, lw=0.5)
    ax[0].set_title(f"Elfouhaily surface, U10 = {U10:.0f} m/s, L = {L:.0f} m, dx = {1000*dx:.1f} mm")
    z = slice(0, int(1.0 / dx))
    ax[1].plot(x[z], h[z], ".-", ms=2, lw=0.8, label="linear")
    ax[1].plot((x + Dx)[z], hc[z], lw=0.8, label="choppy")
    ax[1].set_title("Zoom: first 1 m (dots = grid points)")
    ax[1].legend()
    for a in ax[:2]:
        a.set_xlabel("x [m]"); a.set_ylabel("h [m]")
    ax[2].loglog(kk, kk**3 * Sk)
    for kv, lab in [(2 * k_em, "Bragg (grazing)"), (k_cut, "k_cut"), (kp, "peak")]:
        ax[2].axvline(kv, ls="--", c="gray"); ax[2].text(kv, 2e-3, lab, rotation=90, va="bottom")
    ax[2].set_ylim(1e-4, 5e-2)
    ax[2].set_xlabel("k [rad/m]"); ax[2].set_ylabel("B(k) = k^3 S(k)")
    ax[2].set_title("Curvature spectrum")
    fig.tight_layout()
    fig.savefig("surface_xband.png", dpi=130)
    print("wrote surface_xband.png, surface_xband.npz")
