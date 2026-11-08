"""TEST 1 - Flat PEC plate: the simplest possible case.

What it teaches
  * For zeta = 0 the propagator P is identically zero (both pols): the numerator
    (zeta - zeta') - zeta'(x - x') vanishes. So the MFIE solution is just the Kirchhoff
    current J = J_i (= 2 x incident), i.e. image theory is exact for a flat plate.
  * The tapered beam reflects into a specular lobe at theta_s = +theta_i whose width
    ~ 1/(k g cos theta_i) is set by the taper, not by the surface.
  * Energy conservation for a lossless PEC: int sigma0 d theta_s = 2 pi cos(theta_i).
    This normalization check is the single most useful debugging tool for every later test.
    With the thesis normalization (2.33) (divide by int |psi_i(x,0)|^2 dx) a finite beam
    falls slightly short of 1, by Thorsos' beam correction
        ratio = 1 - (1 + 2 tan^2 theta_i) / (2 (k g cos theta_i)^2),
    which grows toward grazing / for narrow beams. Another face of the taper problem (Ch. 5).
  * Backscatter from a flat plate is (essentially) zero: all clutter comes from roughness.
"""
import numpy as np
import mfie2d as m
from common import plt, check, summary, energy_ratio, savefig

k = 2 * np.pi            # lambda = 1
L, dx = 60.0, 0.1
ts = np.radians(np.linspace(-89.95, 89.95, 7197))

fig, ax = plt.subplots(figsize=(7, 4))
for th_deg, g in [(30, L / 6), (60, L / 6), (60, L / 12)]:
    th = np.radians(th_deg)
    for pol in ["TE", "TM"]:
        print(f"theta_i={th_deg} deg, g={g:.1f} lambda, {pol}")
        sig, d = m.scatter(m.flat_surface(L), dx, k, th, g, pol, ts)
        check("max |P| (should be exactly 0)", np.abs(d["P"]).max(), 0, 1e-14, rel=False)
        pred = 1 - (1 + 2 * np.tan(th) ** 2) / (2 * (k * g * np.cos(th)) ** 2)
        check("energy ratio vs Thorsos beam prediction", energy_ratio(sig, ts, th), pred, 1.5e-3)
        check("specular peak angle [deg]", np.degrees(ts[sig.argmax()]), th_deg, 0.25, rel=False)
        bsc = m.nrcs(d["x"], d["zeta"], 0 * d["x"], d["J"], k, th, g, -th, pol)[0]
        print(f"  backscatter/peak = {10*np.log10(bsc/sig.max()):.0f} dB  (taper sidelobes only)")
        if pol == "TE":
            ax.semilogy(np.degrees(ts), sig, label=f"θi={th_deg}°, g={g:.0f}λ")
ax.set(xlabel="scattering angle θs [deg]", ylabel="σ0 (bistatic)", ylim=(1e-8, None),
       title="Flat plate (TE): specular lobe, width set by the taper")
ax.legend()
savefig(fig, "01_flat_plate.png")
summary()
