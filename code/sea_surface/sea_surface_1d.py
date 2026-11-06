"""Quick-look 1D random sea surfaces: linear Gaussian, choppy (Lagrangian), and time evolution.

Spectrum: JONSWAP (gravity waves only). Dispersion includes capillary term.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

g, sigma_rho = 9.81, 7.4e-5  # gravity [m/s^2], surface tension / density [m^3/s^2]


def jonswap_k(k, Hs=2.0, Tp=8.0, gamma=3.3, dk=None):
    """One-sided JONSWAP wavenumber spectrum S(k) [m^3], scaled so 4*sqrt(var) = Hs."""
    k = np.abs(k)
    w = np.sqrt(g * k)
    wp = 2 * np.pi / Tp
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(w <= wp, 0.07, 0.09)
        r = np.exp(-((w - wp) ** 2) / (2 * s**2 * wp**2))
        Sw = g**2 * w**-5 * np.exp(-1.25 * (wp / w) ** 4) * gamma**r
        Sk = Sw * 0.5 * np.sqrt(g / k)  # S(k) = S(w) dw/dk
    Sk = np.where(k > 0, Sk, 0.0)
    Sk = np.nan_to_num(Sk, posinf=0.0)
    if dk is None:
        dk = k[1] - k[0]
    return Sk * (Hs / 4) ** 2 / np.sum(Sk * dk)


def omega(k):
    k = np.abs(k)
    return np.sqrt(g * k + sigma_rho * k**3)


def spectral_coeffs(N, L, rng, **spec):
    """Hermitian Fourier coefficients h_hat(k) so ifft gives a real surface."""
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    dk = 2 * np.pi / L
    kpos = np.abs(k)
    S = jonswap_k(np.where(k > 0, k, 0.0), dk=dk, **spec)  # energy only on k>0 (waves toward +x)
    # h = 2 Re sum c e^{ikx}  ->  var = sum 4|amp|^2 = sum S dk
    amp = np.sqrt(S * dk / 4)
    c = amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return k, c


def surface(k, c, t=0.0, lam=0.0):
    """Return (x_lagrangian, h). Traveling waves toward +x for k>0 components.

    lam = 0 -> linear Gaussian surface; lam = 1 -> choppy (Gerstner/Tessendorf) displacement.
    """
    N = k.size
    phase = np.exp(-1j * omega(k) * t)
    # one-directional sea: keep k>0 modes, make real by taking 2*Re
    cp = np.where(k > 0, c, 0) * phase
    h = 2 * np.real(np.fft.ifft(cp) * N)
    D = 2 * np.real(np.fft.ifft(-1j * np.sign(k) * cp) * N)  # Hilbert-transform displacement
    return D * lam, h


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, L = 4096, 1000.0
    x = np.linspace(0, L, N, endpoint=False)
    k, c = spectral_coeffs(N, L, rng, Hs=3.0, Tp=9.0)

    Dx0, h_lin = surface(k, c)
    Dx, h = surface(k, c, lam=1.0)
    print(f"Hs from surface: {4 * h_lin.std():.2f} m (target 3.00)")
    print(f"max slope linear: {np.abs(np.gradient(h_lin, x)).max():.3f}")

    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    ax[0].plot(x, h_lin, lw=0.8)
    ax[0].set_title("Linear Gaussian surface (JONSWAP, Hs=3 m, Tp=9 s)")
    sl = slice(0, N // 8)
    ax[1].plot(x[sl], h_lin[sl], lw=1, label="linear")
    ax[1].plot((x + Dx)[sl], h[sl], lw=1, label="choppy")
    ax[1].set_title("Zoom: linear vs choppy (sharper crests, flatter troughs)")
    ax[1].legend()
    kp = k[k > 0]
    ax[2].loglog(kp, jonswap_k(kp, Hs=3.0, Tp=9.0))
    ax[2].set_ylim(1e-8, None)
    ax[2].set_xlabel("k [rad/m]"); ax[2].set_ylabel("S(k)")
    ax[2].set_title("JONSWAP wavenumber spectrum")
    for a in ax[:2]:
        a.set_xlabel("x [m]"); a.set_ylabel("h [m]")
    fig.tight_layout()
    fig.savefig("surfaces_1d.png", dpi=130)

    # time evolution (choppy), first 250 m
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    sl = slice(0, N // 4)
    (ln_lin,) = ax2.plot([], [], lw=0.8, alpha=0.5, label="linear")
    (ln,) = ax2.plot([], [], lw=1.2, label="choppy")
    ax2.set_xlim(0, L / 4); ax2.set_ylim(-4, 4)
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("h [m]"); ax2.legend(loc="upper right")
    ttl = ax2.set_title("")

    def update(i):
        t = i * 0.25
        _, hl = surface(k, c, t)
        d, hc = surface(k, c, t, lam=1.0)
        ln_lin.set_data(x[sl], hl[sl])
        ln.set_data((x + d)[sl], hc[sl])
        ttl.set_text(f"t = {t:5.2f} s")
        return ln, ln_lin, ttl

    FuncAnimation(fig2, update, frames=120, blit=True).save(
        "surface_evolution.gif", writer=PillowWriter(fps=20))
    print("wrote surfaces_1d.png, surface_evolution.gif")
