"""GPU (PyTorch, complex128) version of fastsolve: propagator build, MOMI, far field.

Why: choppy surfaces need dx = lambda/20 or finer (N >= 8800); the CPU Bessel evaluation
then dominates. Same equations as mfie2d; validated against the CPU path in gpu_check.py.

Accuracy note: torch.special.bessel_j1/y1 are only ~1e-6 accurate for 1 <= z <= 25
(exact to round-off elsewhere). So every matrix entry with |x_m - x_n| < 25/k (which
contains every entry with kR < 25) is computed on the CPU with scipy and patched in.
"""
import numpy as np
import torch
from scipy.special import j1, y1

DEV = "cuda"
NEAR_KR = 25.0


def _hankel_band(x, z, k, B):
    """scipy H1^(2)(kR)/R for |m - n| <= B, as an (N, 2B+1) array (nan outside the matrix)."""
    N = len(x)
    off = np.arange(-B, B + 1)
    cols = np.arange(N)[:, None] + off[None, :]
    valid = (cols >= 0) & (cols < N)
    cc = np.clip(cols, 0, N - 1)
    R = np.hypot(x[:, None] - x[cc], z[:, None] - z[cc])
    R[:, B] = 1.0
    H = (j1(k * R) - 1j * y1(k * R)) / R
    return H, cols, valid


def build_A(x, z, zx, zxx, k, pol, block=512):
    """A = I - P on the GPU (complex128) for one polarization."""
    N = len(x)
    dx = x[1] - x[0]
    B = int(np.ceil(NEAR_KR / k / dx)) + 1
    Hb, cols, valid = _hankel_band(x, z, k, B)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=DEV)
    xt, zt, zxt, zxxt = t(x), t(z), t(zx), t(zxx)
    st = torch.sqrt(1 + zxt**2)
    Hb_t, cols_t, valid_t = t(Hb), t(cols), t(valid)
    A = torch.empty((N, N), dtype=torch.complex128, device=DEV)
    c0 = -0.5j * k * dx
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        X = xt[i0:i1, None] - xt[None, :]
        Z = zt[i0:i1, None] - zt[None, :]
        R = torch.hypot(X, Z)
        ii = torch.arange(i0, i1, device=DEV)
        R[ii - i0, ii] = 1.0
        kR = k * R
        H = torch.complex(torch.special.bessel_j1(kR), -torch.special.bessel_y1(kR)) / R
        del kR
        # patch the near band with exact values
        rows = (ii - i0)[:, None].expand(-1, cols_t.shape[1])
        v = valid_t[i0:i1]
        H[rows[v], cols_t[i0:i1][v]] = Hb_t[i0:i1][v]
        if pol == "TE":
            blk = (c0 * H) * (Z - zxt[i0:i1, None] * X) * (st[None, :] / st[i0:i1, None])
            d = -zxxt[i0:i1] * dx / (2 * np.pi * st[i0:i1] ** 2)
        else:
            blk = (c0 * H) * (Z - zxt[None, :] * X)
            d = +zxxt[i0:i1] * dx / (2 * np.pi * st[i0:i1] ** 2)
        blk[ii - i0, ii] = d.to(torch.complex128)
        blk.neg_()
        blk[ii - i0, ii] += 1.0
        A[i0:i1] = blk
        del X, Z, R, H, blk
    return A


def _tril_mv(A, v, block=2048):
    """tril(A) @ v (incl. diagonal) without forming tril(A)."""
    N = A.shape[0]
    y = torch.empty_like(v)
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        y[i0:i1] = A[i0:i1, :i0] @ v[:i0] + torch.tril(A[i0:i1, i0:i1]) @ v[i0:i1]
    return y


def _unit_triu_mv(A, v, block=2048):
    """(I + strict_triu(A)) @ v without forming it."""
    N = A.shape[0]
    y = torch.empty_like(v)
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        y[i0:i1] = v[i0:i1] + torch.triu(A[i0:i1, i0:i1], 1) @ v[i0:i1] + A[i0:i1, i1:] @ v[i1:]
    return y


def momi(A, Ji, n_iter=3):
    """MOMI on A = I - P (GPU). Ji: numpy vector. Returns list of numpy currents."""
    b = torch.as_tensor(Ji, device=DEV).reshape(-1, 1)

    def apply_inv(r):
        y = torch.linalg.solve_triangular(A, r, upper=False)
        return torch.linalg.solve_triangular(A, y, upper=True, unitriangular=True)

    Lv = lambda v: v - _tril_mv(A, v)
    Uv = lambda v: v - _unit_triu_mv(A, v)
    JB = apply_inv(b)
    out, J = [JB], JB
    for _ in range(n_iter):
        J = JB + apply_inv(Lv(Uv(J)))
        out.append(J)
    return [o.reshape(-1).cpu().numpy() for o in out]


def solve_direct(A, Ji):
    return torch.linalg.solve(A, torch.as_tensor(Ji, device=DEV)).cpu().numpy()
