"""Minimal 2-D PEC rough-surface scattering solver following Toporkov (1998), Ch. 2-5.

Everything here is written to map one-to-one onto the thesis equations so you can read
the code next to the PDF. Conventions are the THESIS conventions (not the proposal's):

  * time dependence e^{+j w t}  ->  Green's function G = H0^(2)(k R) / (4j)       (2.8)
    (note 1/(4j) = -j/4; with this sign (laplacian + k^2) G = -delta)
    (the proposal uses e^{-i w t} and H0^(1); the physics is identical, all phases conjugate)
  * surface z = zeta(x), z axis up, angles measured from the vertical (z) axis
  * incident field travels toward +x and down; specular is theta_s = +theta_i,
    backscatter is theta_s = -theta_i                                           (2.34)
  * TE / horizontal pol (HH): psi = E_y, Dirichlet BC, unknown J = dE/dn on surface
  * TM / vertical pol (VV):   psi = H_y, Neumann BC,   unknown J = H on surface

The integral equation (MFIE, eq 2.13) is   J(x) = J_i(x) + int P(x, x') J(x') dx'
which after sampling on a uniform grid x_m becomes  J = J_i + P J              (2.18)
"""
import numpy as np
from scipy.special import hankel2
from scipy.linalg import solve_triangular

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------------------
class Surface:
    """A surface defined by analytic callables for zeta, zeta', zeta''.

    Keeping the surface as a function (not samples) means the *same* realization can be
    re-sampled at any dx -- exactly what Toporkov does in Ch. 4 to study sampling effects.
    """

    def __init__(self, f, fx, fxx, L):
        self.f, self.fx, self.fxx, self.L = f, fx, fxx, L

    def sample(self, dx):
        N = int(round(self.L / dx))
        x = (np.arange(N) - N / 2) * dx  # eq. after (4.14): x_m = (m - N/2) dx
        return x, self.f(x), self.fx(x), self.fxx(x)


def flat_surface(L):
    z = lambda x: np.zeros_like(x)
    return Surface(z, z, z, L)


def sinusoid_surface(L, h, Lambda):
    """zeta = h cos(K x), K = 2 pi / Lambda. Rayleigh's original rough surface (1896)."""
    K = 2 * np.pi / Lambda
    return Surface(lambda x: h * np.cos(K * x),
                   lambda x: -h * K * np.sin(K * x),
                   lambda x: -h * K**2 * np.cos(K * x), L)


def pm_spectrum(K, U, alpha=8.1e-3, beta=0.74, g=9.81):
    """1-D Pierson-Moskowitz spectrum, thesis eq (4.2). Two-sided in K, units m^3."""
    K = np.abs(K)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W = alpha / (4 * K**3) * np.exp(-beta * g**2 / (K**2 * U**4))
    return np.where(K > 0, W, 0.0)


def pm_rms_height(U, alpha=8.1e-3, beta=0.74, g=9.81):
    """Closed form of sqrt(int W dK) for (4.2): <zeta^2> = alpha U^4 / (4 beta g^2)  (4.7)."""
    return np.sqrt(alpha * U**4 / (4 * beta * g**2))


def random_surface(L, spectrum, Kc, rng):
    """Spectral method (thesis Ch. 3): Gaussian random Fourier amplitudes with variance
    W(K_m) dK, periodic on [−L/2, L/2), band-limited to |K| <= Kc.

    zeta(x) = 2 Re sum_{m>0} c_m exp(j K_m x),  E|c_m|^2 = W(K_m) dK,
    so Var[zeta] = sum_{m != 0} W(K_m) dK  ~=  int W dK.
    Derivatives come for free: multiply c_m by (jK_m) and (jK_m)^2.
    """
    dK = 2 * np.pi / L
    K = dK * np.arange(1, int(Kc / dK) + 1)
    v = spectrum(K) * dK
    c = np.sqrt(v / 2) * (rng.standard_normal(K.size) + 1j * rng.standard_normal(K.size))

    def make(power):
        def fn(x):
            ph = np.exp(1j * np.outer(x, K))
            return 2 * np.real(ph @ (c * (1j * K) ** power))
        return fn

    s = Surface(make(0), make(1), make(2), L)
    s.K, s.c = K, c
    return s


# --------------------------------------------------------------------------------------
# Incident field: plane wave with the Thorsos taper (4.12) / (5.3)
# --------------------------------------------------------------------------------------
def thorsos_field(x, z, k, theta_i, g):
    """Returns psi_i and its gradient (dpsi/dx, dpsi/dz). Analytic derivative of (4.12)."""
    st, ct, tt = np.sin(theta_i), np.cos(theta_i), np.tan(theta_i)
    a = x + z * tt                                   # distance across the beam
    q = (k * g * ct) ** 2
    w = (2 * a**2 / g**2 - 1) / q                    # Thorsos' correction w(x,z)
    phi = x * st - z * ct                            # plane-wave phase
    psi = np.exp(-1j * k * phi * (1 + w)) * np.exp(-a**2 / g**2)
    dw = 4 * a / (g**2 * q)                          # dw/da
    dpsi_dx = psi * (-1j * k * (st * (1 + w) + phi * dw) - 2 * a / g**2)
    dpsi_dz = psi * (-1j * k * (-ct * (1 + w) + phi * dw * tt) - 2 * a * tt / g**2)
    return psi, dpsi_dx, dpsi_dz


def exact_beam_field(x, z, k, theta_i, g, n_kappa=801):
    """An EXACT Helmholtz solution with a Gaussian footprint: angular-spectrum superposition
    of propagating plane waves  psi = int A(kappa) exp(-j(kappa x - gamma z)) dkappa,
    gamma = sqrt(k^2 - kappa^2), A Gaussian around k sin(theta_i) (footprint ~ g / cos).
    Unlike the Thorsos taper it has zero Helmholtz residual, so EFIE and MFIE must agree.
    Normalised so psi(0,0) ~ 1 like the taper."""
    G = g / np.cos(theta_i)
    k0 = k * np.sin(theta_i)
    kap = np.linspace(max(-k, k0 - 8 / G), min(k, k0 + 8 / G), n_kappa)[1:-1]
    gam = np.sqrt(k**2 - kap**2)
    A = np.exp(-((kap - k0) * G / 2) ** 2)
    A = A / np.trapezoid(A, kap) * (kap[1] - kap[0])
    E = np.exp(-1j * (np.multiply.outer(x, kap) - np.multiply.outer(z, gam)))
    psi = E @ A
    return psi, E @ (-1j * kap * A), E @ (1j * gam * A)


def kirchhoff_current(x, zeta, zx, k, theta_i, g, pol, field=thorsos_field):
    """Forcing term J_i: (2.14a) TM: 2 H_i ; (2.15a) TE: 2 dE_i/dn (unit normal, upward)."""
    psi, px, pz = field(x, zeta, k, theta_i, g)
    if pol == "TM":
        return 2 * psi
    s = np.sqrt(1 + zx**2)
    return 2 * (-zx * px + pz) / s


def kapp_min_waist(k, theta_i, A=9.4):
    """Minimum taper parameter g (thesis eq 5.8, A = 9.4 found for P-M surfaces)."""
    return A / ((np.pi / 2 - theta_i) * k * np.cos(theta_i))


# --------------------------------------------------------------------------------------
# Propagator matrix (2.14b)/(2.15b) with (2.16)/(2.17), and the curvature diagonal (4.1)
# --------------------------------------------------------------------------------------
def propagator(x, zeta, zx, zxx, k, pol, curvature=True):
    """Discrete MFIE propagator P_mn = P(x_m, x_n) dx.

    TE: P = -2 dG/dn(obs) * sqrt(1+zeta'(x')^2)
        = -(jk/2) H1(kR) [ (zeta-zeta') - zeta'(x)(x-x') ] / R * s(x')/s(x)
    TM: P = +2 dG/dn'(src) * sqrt(1+zeta'(x')^2)
        = -(jk/2) H1(kR) [ (zeta-zeta') - zeta'(x')(x-x') ] / R
    On the diagonal the kernel is finite; its limit is the 'curvature term'
        P_mm = +/- zeta''(x_m) dx / (2 pi (1 + zeta'(x_m)^2))        (4.1)
    with + for TM and - for TE, as in the thesis. test_03 checks this against an
    independent EFIE solve.
    """
    dx = x[1] - x[0]
    X = x[:, None] - x[None, :]
    Z = zeta[:, None] - zeta[None, :]
    R = np.hypot(X, Z)
    np.fill_diagonal(R, 1.0)                         # avoid 0/0; diagonal overwritten below
    H1 = hankel2(1, k * R)
    s = np.sqrt(1 + zx**2)
    if pol == "TE":
        P = -0.5j * k * H1 * (Z - zx[:, None] * X) / R * (s[None, :] / s[:, None]) * dx
        diag = -zxx * dx / (2 * np.pi * s**2)
    else:
        P = -0.5j * k * H1 * (Z - zx[None, :] * X) / R * dx
        diag = +zxx * dx / (2 * np.pi * s**2)
    np.fill_diagonal(P, diag if curvature else 0.0)
    return P


def efie_matrix_TE(x, zeta, zx, k):
    """First-kind EFIE (2.10) for TE:  E_i(x_m) = sum_n G(x_m, x_n) J_n s_n dx.
    Log-singular diagonal integrated analytically over the cell (small-argument H0)."""
    dx = x[1] - x[0]
    X = x[:, None] - x[None, :]
    Z = zeta[:, None] - zeta[None, :]
    R = np.hypot(X, Z)
    np.fill_diagonal(R, 1.0)
    s = np.sqrt(1 + zx**2)
    A = -0.25j * hankel2(0, k * R) * s[None, :] * dx
    h = s * dx                                        # arc length of the cell
    self_term = -0.25j * h * (1 - 2j / np.pi * (np.log(EULER_GAMMA * k * h / 4) - 1))
    np.fill_diagonal(A, self_term)
    return A


# --------------------------------------------------------------------------------------
# Solvers: direct (2.20), Neumann series (2.21), MOMI (2.22)-(2.25)
# --------------------------------------------------------------------------------------
def solve_direct(P, Ji):
    return np.linalg.solve(np.eye(len(Ji)) - P, Ji)


def solve_neumann(P, Ji, n_iter):
    """J = Ji + P Ji + P^2 Ji + ...  Returns list of iterates (may diverge!)."""
    J, out = Ji.copy(), []
    for _ in range(n_iter):
        J = Ji + P @ J
        out.append(J.copy())
    return out


def solve_momi(P, Ji, n_iter):
    """Method of Ordered Multiple Interactions (Kapp & Brown 1996), thesis (2.23)-(2.25).

    P = L + U, L = points to the LEFT of the observer (incl. diagonal), U = to the RIGHT.
      J_B      = (I-U)^-1 (I-L)^-1 J_i                     'new Born term'  (2.24)
      J^(p+1)  = J_B + (I-U)^-1 (I-L)^-1 L U J^(p)
    Physically: one forward sweep (left->right) then one backward sweep picks up each
    order of multiple scattering, which is why so few iterations are needed.
    Returns [J_B, J^(1), J^(2), ...].
    """
    N = len(Ji)
    I = np.eye(N)
    Lm, Um = np.tril(P), np.triu(P, 1)
    ImL, ImU = I - Lm, I - Um

    def apply_inv(b):
        y = solve_triangular(ImL, b, lower=True)
        return solve_triangular(ImU, y, lower=False)

    JB = apply_inv(Ji)
    out, J = [JB], JB
    for _ in range(n_iter):
        J = JB + apply_inv(Lm @ (Um @ J))
        out.append(J)
    return out


# --------------------------------------------------------------------------------------
# Far field and NRCS (2.30)-(2.34)
# --------------------------------------------------------------------------------------
def nrcs(x, zeta, zx, J, k, theta_i, g, theta_s, pol, field=thorsos_field):
    """Bistatic NRCS sigma0(theta_i, theta_s), eq (2.33), normalised by int |psi_i(x,0)|^2 dx.

    Far zone: H0^(2)(k|r-r'|) ~ sqrt(2/(pi k r)) e^{-j(kr - pi/4)} e^{+jk rhat.r'},
    so  2 pi r |psi_s|^2 = |I|^2 / (4k)  with
      TE: I = int exp(jk(x sin ts + zeta cos ts)) J sqrt(1+zeta'^2) dx           (2.32)
      TM: I = int jk (cos ts - zeta' sin ts) exp(jk(...)) J dx                   (2.31)
    """
    dx = x[1] - x[0]
    ts = np.atleast_1d(theta_s)[:, None]
    ph = np.exp(1j * k * (x[None, :] * np.sin(ts) + zeta[None, :] * np.cos(ts)))
    if pol == "TE":
        I = ph @ (J * np.sqrt(1 + zx**2)) * dx
    else:
        I = (1j * k * (np.cos(ts) - zx[None, :] * np.sin(ts)) * ph) @ J * dx
    xx = np.linspace(-4 * g / np.cos(theta_i), 4 * g / np.cos(theta_i), 20001)
    if field is thorsos_field:
        inc = np.sum(np.abs(field(xx, 0 * xx, k, theta_i, g)[0]) ** 2) * (xx[1] - xx[0])
    else:  # exact incident power flux through z = 0 per unit |psi|^2 (Poynting, z-component)
        psi, _, pz = field(xx[::10], 0 * xx[::10], k, theta_i, g)
        inc = np.sum(np.imag(np.conj(psi) * pz)) / k * (xx[10] - xx[0]) / np.cos(theta_i)
    return np.abs(I) ** 2 / (4 * k * inc)


def scatter(surface, dx, k, theta_i, g, pol, theta_s, curvature=True, method="direct", n_iter=2,
            field=thorsos_field):
    """Convenience wrapper: sample surface, build P, solve, return (sigma0, extras)."""
    x, z, zx, zxx = surface.sample(dx)
    Ji = kirchhoff_current(x, z, zx, k, theta_i, g, pol, field)
    P = propagator(x, z, zx, zxx, k, pol, curvature)
    J = solve_direct(P, Ji) if method == "direct" else solve_momi(P, Ji, n_iter)[-1]
    sig = nrcs(x, z, zx, J, k, theta_i, g, theta_s, pol, field)
    return sig, dict(x=x, zeta=z, J=J, Ji=Ji, P=P)
