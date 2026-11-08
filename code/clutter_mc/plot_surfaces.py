"""Figure 5: what the solver actually sees. Gaussian vs choppy windows at 5/10/15 m/s
(same seed), the beam footprint, and the fraction of P-M height/slope the window represents."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import surfaces as S
import run_mc as R

fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, U in zip(axs, (5.0, 10.0, 15.0)):
    sea = S.Sea(U, 0, R.LAM, R.L_GEN_LAM["full"] * R.LAM)
    N = R.L_WIN_LAM * 10
    for choppy, c in ((False, "C0"), (True, "C3")):
        w = sea.window(N, R.LAM / 10, choppy)
        ax.plot(w["x"], w["z"], c, lw=0.9, label=f"{'choppy' if choppy else 'Gaussian'} (window h_rms {w['h_rms_win']:.2f} m, "
                f"slope {w['slope_rms_win']:.3f}{', min dx/du %.2f' % w['jac_min_window'] if choppy else ''})")
    ax.axvspan(-R.G, R.G, color="gold", alpha=0.15, label="footprint ±g")
    Kp = np.sqrt(2 * 0.74 / 3) * 9.81 / U**2
    ax.set_title(f"U = {U:g} m/s: P-M h_rms {sea.h_rms_nominal:.2f} m, peak wavelength {2*np.pi/Kp:.0f} m, "
                 f"nominal rms slope {sea.slope_rms_nominal:.3f}", fontsize=10)
    ax.set_ylabel("z [m]")
    ax.legend(fontsize=7, loc="upper right")
axs[-1].set_xlabel("x [m]  (solve window, 8g = 101 m)")
fig.tight_layout()
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"), exist_ok=True)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig5_surfaces.png"), dpi=130)
