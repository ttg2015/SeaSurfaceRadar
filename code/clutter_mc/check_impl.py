"""Implementation checks: fast solver == mfie2d, surface generator == direct Fourier sums,
choppy surfaces have sharp crests (positive skewness). Plus a single-thread timing."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import numpy as np
import surfaces as S
import fastsolve as F
from surfaces import m

lam = 0.23
k = 2 * np.pi / lam
ok = True


def report(name, val, tol):
    global ok
    good = val <= tol
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: {val:.2e} (tol {tol:g})")


print("1) surface generator vs direct Fourier sums")
sea = S.Sea(10.0, 7, lam, L_gen=4096 * lam)
for choppy in [False, True]:
    w = sea.window(1000, lam / 10, choppy)
    xs = w["x"][::25]
    ph = lambda u, p, D=False: 2 * np.real(np.exp(1j * np.outer(u, sea.K)) @ (
        (1j if D else 1) * sea.c * (1j * sea.K) ** p))
    if not choppy:
        zd = ph(xs, 0)
        zxd = ph(xs, 1)
        zxxd = ph(xs, 2)
    else:
        u = xs.copy()
        for _ in range(50):
            u -= (u + ph(u, 0, True) - xs) / (1 + ph(u, 1, True))
        J1 = 1 + ph(u, 1, True)
        zd, h1, h2, D2 = ph(u, 0), ph(u, 1), ph(u, 2), ph(u, 2, True)
        zxd, zxxd = h1 / J1, (h2 * J1 - h1 * D2) / J1**3
    zfull = w["z"] + (None or 0)
    off = np.mean(w["z"][::25] - zd)  # mean removed in window(); compare up to a constant
    lbl = "choppy" if choppy else "gauss "
    report(f"{lbl} z     max err / h_rms", np.max(abs(w["z"][::25] - zd - off)) / sea.h_rms_nominal, 1e-6)
    report(f"{lbl} z'    max err / slope", np.max(abs(w["zx"][::25] - zxd)) / sea.slope_rms_nominal, 1e-4)
    report(f"{lbl} z''   max err / max|z''|", np.max(abs(w["zxx"][::25] - zxxd)) / np.max(abs(zxxd)), 1e-3)
    print(f"     skewness {w['skew_win']:+.3f}, min dx/du window {w['jac_min_window']:.3f}")

print("2) fast propagator / MOMI vs mfie2d")
w = S.Sea(10.0, 3, lam, 4096 * lam).window(800, lam / 10, True)
x, z, zx, zxx = w["x"], w["z"], w["zx"], w["zxx"]
A = F.build_A(x, z, zx, zxx, k)
th, g = np.radians(80), 15 * lam
for pol in ["TE", "TM"]:
    P = m.propagator(x, z, zx, zxx, k, pol)
    report(f"{pol} max|A - (I-P)| / max|P|", np.max(abs(A[pol] - (np.eye(len(x)) - P))) / abs(P).max(), 1e-12)
    Ji = m.kirchhoff_current(x, z, zx, k, th, g, pol)
    Jref = m.solve_momi(P, Ji, 3)
    Jf = F.momi(A[pol], Ji, 3)
    report(f"{pol} MOMI fast vs mfie2d (rel)", max(np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in zip(Jref, Jf)), 1e-10)
    Jd = m.solve_direct(P, Ji)
    print(f"     MOMI(2) vs direct: {np.linalg.norm(Jf[2]-Jd)/np.linalg.norm(Jd):.1e}, "
          f"MOMI(3): {np.linalg.norm(Jf[3]-Jd)/np.linalg.norm(Jd):.1e}")

print("3) single-thread timing at production size")
N = 4400
sea = S.Sea(15.0, 1, lam, 4096 * lam)
t = time.time(); w = sea.window(N, lam / 10, True); t_s = time.time() - t
t = time.time(); A = F.build_A(w["x"], w["z"], w["zx"], w["zxx"], k); t_b = time.time() - t
Ji = m.kirchhoff_current(w["x"], w["z"], w["zx"], k, th, 55 * lam, "TE")
t = time.time(); F.momi(A["TE"], Ji, 3); t_m = time.time() - t
print(f"  N={N}: surface {t_s:.2f}s, build TE+TM {t_b:.1f}s, MOMI(3) one RHS {t_m:.2f}s")
print("ALL PASS" if ok else "SOME FAILED")
