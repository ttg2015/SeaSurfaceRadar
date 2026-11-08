"""GPU path vs CPU path on several realizations (same dx): sigma must agree to << 0.01 dB."""
import numpy as np, time
import run_mc as R
worst = 0
for U, seed in [(10.0, 9001), (15.0, 9002), (5.0, 9003), (15.0, 9003)]:
    t = time.time(); c = R.solve_realization(U, seed, backend="cpu"); tc = time.time() - t
    t = time.time(); g = R.solve_realization(U, seed, backend="gpu", direct=True); tg = time.time() - t
    for a, b in zip(c, g):
        d = abs(10 * np.log10(a["sigma"] / b["sigma"]))
        dd = abs(10 * np.log10(b["sigma"] / b["sigma_direct"]))
        de = abs(a["energy"] - b["energy"])
        worst = max(worst, d)
        print(f"U={U} {seed} {a['surf']:6s} {a['pol']} {a['theta']}: cpu-gpu {d:.1e} dB, gpu MOMI-direct {dd:.1e} dB, energy diff {de:.1e}")
    print(f"   time cpu {tc:.1f}s  gpu {tg:.1f}s")
print("WORST cpu-gpu dB:", worst, "PASS" if worst < 1e-3 else "FAIL")
