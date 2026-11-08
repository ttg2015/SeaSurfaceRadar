"""Memory- and time-lean versions of mfie2d's propagator and MOMI. Same equations; validated
against mfie2d in validate.py (agreement to round-off).

Speed-ups:
  * H1^(2)(kR) = J1(kR) - j Y1(kR) via scipy's real-argument Bessels (~2x faster than hankel2)
  * TE and TM matrices built in one row-blocked pass that shares the Bessel evaluations
  * we store A = I - P only (one N x N matrix per polarization). MOMI needs
        (I-L)^{-1}: lower triangle of A (its diagonal is 1 - P_mm)
        (I-U)^{-1}: strict upper triangle of A with a unit diagonal
        L v = v - tril(A) v,   U v = v - unit_triu(A) v
    all via BLAS trmv / LAPACK trtrs on A without copies.
"""
import numpy as np
from scipy.special import j1, y1
from scipy.linalg import solve_triangular, get_blas_funcs


def build_A(x, z, zx, zxx, k, pols=("TE", "TM"), block=256):
    N = len(x)
    dx = x[1] - x[0]
    s = np.sqrt(1 + zx**2)
    out = {p: np.empty((N, N), complex) for p in pols}
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        X = x[i0:i1, None] - x[None, :]
        Z = z[i0:i1, None] - z[None, :]
        R = np.hypot(X, Z)
        ii = np.arange(i0, i1)
        R[ii - i0, ii] = 1.0
        kR = k * R
        common = (-0.5j * k * dx) * (j1(kR) - 1j * y1(kR)) / R
        if "TE" in pols:
            blk = common * (Z - zx[i0:i1, None] * X) * (s[None, :] / s[i0:i1, None])
            blk[ii - i0, ii] = -zxx[i0:i1] * dx / (2 * np.pi * s[i0:i1] ** 2)
            np.negative(blk, out=blk)
            blk[ii - i0, ii] += 1.0
            out["TE"][i0:i1] = blk
        if "TM" in pols:
            blk = common * (Z - zx[None, :] * X)
            blk[ii - i0, ii] = +zxx[i0:i1] * dx / (2 * np.pi * s[i0:i1] ** 2)
            np.negative(blk, out=blk)
            blk[ii - i0, ii] += 1.0
            out["TM"][i0:i1] = blk
    return out


def momi(A, Ji, n_iter=3):
    """MOMI on A = I - P. Returns list [J_B, J^(1), ..., J^(n_iter)] like mfie2d.solve_momi."""
    AT = A.T  # Fortran-ordered view, no copy
    trmv = get_blas_funcs("trmv", (AT,))

    def apply_inv(b):
        y = solve_triangular(A, b, lower=True, check_finite=False)
        return solve_triangular(A, y, lower=False, unit_diagonal=True, check_finite=False)

    def Lv(v):   # tril(P) v = v - tril(A) v ;  tril(A) = (triu(A^T))^T
        return v - trmv(AT, v, lower=0, trans=1)

    def Uv(v):   # strict_triu(P) v = v - unit_triu(A) v
        return v - trmv(AT, v, lower=1, trans=1, diag=1)

    JB = apply_inv(Ji)
    out, J = [JB], JB
    for _ in range(n_iter):
        J = JB + apply_inv(Lv(Uv(J)))
        out.append(J)
    return out
