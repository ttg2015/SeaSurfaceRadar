"""Shared helpers for the test scripts."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

_results = []


def check(name, value, target, tol, rel=True):
    err = abs(value - target) / (abs(target) if rel and target != 0 else 1)
    ok = err <= tol
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {value:.6g}, expected {target:.6g} "
          f"({'rel' if rel else 'abs'} err {err:.2e}, tol {tol:g})")
    return ok


def summary():
    n = len(_results)
    print(f"\n{sum(_results)}/{n} checks passed")
    return all(_results)


def energy_ratio(sig, ts, theta_i):
    """Lossless PEC: int sigma0 d(theta_s) = 2 pi cos(theta_i)  (power in = power out)."""
    return np.trapezoid(sig, ts) / (2 * np.pi * np.cos(theta_i))


def savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path}")
