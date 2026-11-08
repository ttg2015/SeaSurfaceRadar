"""Gaussian and choppy (Lagrangian / Gerstner) 1-D Pierson-Moskowitz surfaces on a long
periodic domain, evaluated on a shorter solve window.

Why a long domain: at U = 10-15 m/s the P-M peak wavelength (91 m, 205 m) is longer than any
affordable solve window. Generating on L_gen >> window keeps the long waves in the sample, so
inside the window they show up as tilt, offset and gentle curvature -- as on the real sea.
The solve cost depends only on the window.

Linear (Gaussian) surface, same convention as mfie2d.random_surface:
    h(u) = 2 Re sum_m c_m exp(j K_m u),  E|c_m|^2 = W(K_m) dK,  K_m = m dK,  dK = 2 pi / L_gen
Choppy surface: the same h, displaced horizontally (Gerstner / "choppy wave" model):
    x(u) = u + D(u),  z = h(u),  D(u) = 2 Re sum_m j c_m exp(j K_m u)
For h = a cos(Ku) this gives x = u - a sin(Ku): points crowd together at crests (sharp crests)
and spread out in troughs (flat troughs). The surface stays single-valued while
dx/du = 1 + D'(u) > 0; where it is <= 0 the crest has overturned.

Both share the same random coefficients, so Gaussian vs choppy comparisons are paired.
"""
import numpy as np
from scipy.interpolate import CubicSpline

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "thesis_tests"))
import mfie2d as m  # noqa: E402


class Sea:
    def __init__(self, U, seed, lam, L_gen, Kc_over_k=4.99, du_over_lam=1 / 80):
        self.U, self.seed, self.lam = U, seed, lam
        self.k = 2 * np.pi / lam
        self.L_gen = L_gen
        self.dK = 2 * np.pi / L_gen
        self.M = int(Kc_over_k * self.k / self.dK)
        self.K = self.dK * np.arange(1, self.M + 1)
        W = m.pm_spectrum(self.K, U)
        rng = np.random.default_rng(seed)
        n1, n2 = rng.standard_normal(self.M), rng.standard_normal(self.M)
        self.c = np.sqrt(W * self.dK / 2) * (n1 + 1j * n2)
        # nominal (ensemble) statistics of what this discretisation represents
        self.h_rms_nominal = np.sqrt(2 * np.sum(W) * self.dK)
        self.slope_rms_nominal = np.sqrt(2 * np.sum(self.K**2 * W) * self.dK)
        # fine u-grid for FFT evaluation
        n = L_gen / (du_over_lam * lam)
        if abs(n - round(n)) > 1e-6 or round(n) % 2:
            raise ValueError("L_gen must be an even multiple of du (use L_gen = integer * lam)")
        self.n = int(round(n))
        self.du = du_over_lam * lam
        self.u = (np.arange(self.n) - self.n // 2) * self.du
        self._cache = {}

    def _grid(self, which, p):
        """which = 'h' or 'D'; p = derivative order. Values on the fine u grid (exact)."""
        key = (which, p)
        if key not in self._cache:
            a = np.zeros(self.n, complex)
            coef = self.c * (1j * self.K) ** p
            if which == "D":
                coef = 1j * coef
            idx = np.arange(1, self.M + 1)
            # u_j = (j - n/2) du  ->  exp(j K_m u_j) = exp(2 pi j m j/n) * (-1)^m
            a[idx] = coef * (-1.0) ** idx
            self._cache[key] = 2 * np.real(np.fft.ifft(a) * self.n)
        return self._cache[key]

    # ------------------------------------------------------------------------------------
    def window(self, N, dx, choppy, newton_tol=1e-11):
        """Surface on x_m = (m - N/2) dx. Returns dict with x, z, zx, zxx and diagnostics.
        z has its window mean removed (a vertical shift: keeps the tapered footprint centred
        on the window; statistically harmless because the surface is stationary)."""
        x = (np.arange(N) - N / 2) * dx
        if not choppy:
            r = dx / self.du
            ri = int(round(r))
            if abs(r - ri) > 1e-9:
                raise ValueError("dx must be a multiple of the generation grid du")
            j = self.n // 2 + (np.arange(N) - N // 2) * ri
            z, zx, zxx = (self._grid("h", p)[j] for p in (0, 1, 2))
            jac_min_win = 1.0
        else:
            # spline the needed u range (window +- max displacement + margin)
            D0 = self._grid("D", 0)
            pad = np.max(np.abs(D0)) + 20 * self.lam
            sel = np.nonzero((self.u > x[0] - pad) & (self.u < x[-1] + pad))[0]
            uu = self.u[sel]
            sp = {key: CubicSpline(uu, self._grid(*key)[sel]) for key in
                  [("h", 0), ("h", 1), ("h", 2), ("D", 0), ("D", 1), ("D", 2)]}
            jac = 1 + self._grid("D", 1)[sel]
            u = x - sp[("D", 0)](x)
            for _ in range(60):
                res = u + sp[("D", 0)](u) - x
                u = u - res / (1 + sp[("D", 1)](u))
                if np.max(np.abs(res)) < newton_tol:
                    break
            else:
                raise RuntimeError("Newton inversion x(u) did not converge (overturning?)")
            h, h1, h2 = (sp[("h", p)](u) for p in (0, 1, 2))
            J1 = 1 + sp[("D", 1)](u)
            D2 = sp[("D", 2)](u)
            z = h
            zx = h1 / J1
            zxx = (h2 * J1 - h1 * D2) / J1**3
            inwin = (uu > u[0]) & (uu < u[-1])
            jac_min_win = float(jac[inwin].min())
        jac_all = 1 + self._grid("D", 1)
        z = z - z.mean()
        return dict(x=x, z=z, zx=zx, zxx=zxx,
                    jac_min_window=jac_min_win,          # min dx/du inside the window
                    jac_min_domain=float(jac_all.min()),  # min dx/du on the whole L_gen
                    h_rms_win=float(z.std()), slope_rms_win=float(np.sqrt(np.mean(zx**2))),
                    skew_win=float(np.mean(z**3) / z.std() ** 3))
