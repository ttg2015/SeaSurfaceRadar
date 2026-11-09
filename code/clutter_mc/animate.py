"""Animation: an X-band-ish (lambda = 23 cm) radar beam at 80 deg (10 deg grazing) on a moving
choppy sea, U = 10 m/s, waves travelling toward the radar. Every frame is a real MFIE solve
(lambda/40, curvature term, MOMI on the GPU), so the clutter fluctuations are the physics, not a cartoon.

Panels per frame
  1. the whole 101 m solve window: surface, beam footprint, induced HH current |J| on the surface
  2. zoom on the footprint: total HH field Re{(psi_inc + psi_scat) e^{j phi}} computed from the solved
     current; phi advances each frame so the wavefronts visibly move (radar phase slowed for display)
  3. HH and VV backscatter sigma0 vs sea time

usage: python animate.py [n_frames] [dt_seconds] [seed]
Frames go to figures/anim_frames/, per-frame numbers to results/animation.jsonl (resumable).
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

import surfaces as S
import gpusolve as GS
import run_mc as R
from surfaces import m

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 480
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.025
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
U, TH, DXDIV = 10.0, np.radians(80), 40
LAM, K, G = R.LAM, R.K, R.G
N = R.L_WIN_LAM * DXDIV
DX = LAM / DXDIV
HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "figures", "anim_frames")
os.makedirs(FR, exist_ok=True)
LOG = os.path.join(HERE, "results", "animation.jsonl")

sea0 = S.Sea(U, SEED, LAM, R.L_GEN_LAM["full"] * LAM)
z_off = sea0.window(N, DX, True, demean=False)["z"].mean()   # fixed offset: the footprint may wander
XZ = (-18.0, 18.0)


def surface(t):
    w = sea0.at_time(t).window(N, DX, True, demean=False)
    w["z"] = w["z"] - z_off
    return w


def scat_TE(xp, zp, x, z, zx, J, chunk=1500):
    """psi_s(r) = +(j/4) sum_n H0^(2)(k|r - r_n|) J_n sqrt(1+zx_n^2) dx   (TE, single layer; sign checked)."""
    xs, zs = torch.as_tensor(x, device="cuda"), torch.as_tensor(z, device="cuda")
    w = torch.as_tensor(J * np.sqrt(1 + zx**2) * DX * 0.25j, device="cuda")
    out = np.empty(len(xp), complex)
    for i0 in range(0, len(xp), chunk):
        X = torch.as_tensor(xp[i0:i0 + chunk], device="cuda")[:, None] - xs[None, :]
        Z = torch.as_tensor(zp[i0:i0 + chunk], device="cuda")[:, None] - zs[None, :]
        kR = K * torch.hypot(X, Z)
        H0 = torch.complex(torch.special.bessel_j0(kR), -torch.special.bessel_y0(kR))
        out[i0:i0 + chunk] = (H0 @ w).cpu().numpy()
    return out


def solve(w, pol):
    x, z, zx, zxx = w["x"], w["z"], w["zx"], w["zxx"]
    A = GS.build_A(x, z, zx, zxx, K, pol)
    Ji = m.kirchhoff_current(x, z, zx, K, TH, G, pol)
    Js = GS.momi(A, Ji, 4)
    del A
    torch.cuda.empty_cache()
    conv = np.linalg.norm(Js[-1] - Js[-2]) / np.linalg.norm(Js[-1])
    sig = m.nrcs(x, z, zx, Js[-1], K, TH, G, -TH, pol)[0]
    return Js[-1], sig, conv


# ---------------------------------------------------------------- sign / accuracy check
w = surface(0.0)
J, s0, conv = solve(w, "TE")
sel = np.nonzero(np.abs(w["x"]) < 8)[0][::200]
nx, nz = -w["zx"][sel], np.ones(len(sel))
nn = np.hypot(nx, nz)
d = 0.002
xp, zp = w["x"][sel] + d * nx / nn, w["z"][sel] + d * nz / nn
pi = np.asarray(m.thorsos_field(xp, zp, K, TH, G)[0])
ps = scat_TE(xp, zp, w["x"], w["z"], w["zx"], J)
r_plus = np.median(np.abs(pi + ps) / np.abs(pi))
r_minus = np.median(np.abs(pi - ps) / np.abs(pi))
print(f"sign check, |psi_total|/|psi_inc| 2 mm above the PEC surface: +sign {r_plus:.3f}, -sign {r_minus:.3f} "
      f"(expect ~k*d = {K*d:.3f} for the right sign)", flush=True)
SIGN = 1.0 if r_plus < r_minus else -1.0
assert min(r_plus, r_minus) < 0.2, "scattered-field formula failed the boundary-condition check"

# ---------------------------------------------------------------- field grid (fixed)
zall = w["z"][(w["x"] > XZ[0]) & (w["x"] < XZ[1])]
ZR = (zall.min() - 0.6, zall.max() + 1.6)
h = LAM / 6
gx = np.arange(XZ[0], XZ[1], h)
gz = np.arange(ZR[0], ZR[1], h)
GX, GZ = np.meshgrid(gx, gz)
print(f"field grid {len(gx)} x {len(gz)} = {GX.size} points; z range {ZR[0]:.2f}..{ZR[1]:.2f} m", flush=True)
XC = (-1.5, 1.5)
cx = np.arange(XC[0], XC[1], LAM / 14)
cz_rel = np.arange(-0.35, 0.85, LAM / 14)

done = {}
if os.path.exists(LOG):
    for l in open(LOG):
        r = json.loads(l); done[r["frame"]] = r
log = open(LOG, "a")
for f in range(NF):
    png = os.path.join(FR, f"frame_{f:04d}.png")
    if f in done and os.path.exists(png):
        continue
    t0 = time.time()
    t = f * DT
    w = surface(t)
    J, sHH, cHH = solve(w, "TE")
    _, sVV, cVV = solve(w, "TM")
    rec = dict(frame=f, t=t, seed=SEED, sigma_HH=float(sHH), sigma_VV=float(sVV), conv_HH=float(cHH),
               conv_VV=float(cVV), jac_min=w["jac_min_window"])
    # field on the wide grid and on a true-aspect close-up (points under the surface masked)
    def field(GXa, GZa):
        zsurf = np.interp(GXa[0], w["x"], w["z"])
        above = GZa > zsurf[None, :]
        Pa = np.full(GXa.shape, np.nan, complex)
        xp, zp = GXa[above], GZa[above]
        Pa[above] = m.thorsos_field(xp, zp, K, TH, G)[0] + SIGN * scat_TE(xp, zp, w["x"], w["z"], w["zx"], J)
        return Pa, zsurf
    P, zs_on_grid = field(GX, GZ)
    zc = np.interp(0.0, w["x"], w["z"])
    CX, CZ = np.meshgrid(cx, zc + cz_rel)
    PC, zs_c = field(CX, CZ)
    done[f] = rec
    log.write(json.dumps(rec) + "\n"); log.flush()

    # ---------------------------------------------------------------- render
    ts = np.array([done[i]["t"] for i in sorted(done) if i <= f])
    hh = 10 * np.log10([done[i]["sigma_HH"] for i in sorted(done) if i <= f])
    vv = 10 * np.log10([done[i]["sigma_VV"] for i in sorted(done) if i <= f])
    fig = plt.figure(figsize=(16, 9), dpi=80, facecolor="#0b1020")
    gsp = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.6, 1.0], width_ratios=[2.3, 1], hspace=0.38, wspace=0.12,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)
    a1, a2, a2c, a3 = (fig.add_subplot(gsp[0, :]), fig.add_subplot(gsp[1, 0]), fig.add_subplot(gsp[1, 1]),
                       fig.add_subplot(gsp[2, :]))
    for ax in (a1, a2, a2c, a3):
        ax.set_facecolor("#0b1020")
        ax.tick_params(colors="0.8", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("0.4")
    fig.suptitle(f"Radar at 10° grazing (θ = 80°, λ = 23 cm) on a choppy 10 m/s sea moving toward the radar   ·   "
                 f"sea time {t:5.2f} s", color="w", fontsize=14)
    # panel 1: whole window
    pts = np.column_stack([w["x"], w["z"]])[::4]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="inferno", norm=LogNorm(1e-3, 50), lw=2)
    lc.set_array(np.abs(J[::4][:-1]))
    a1.add_collection(lc)
    a1.fill_between(w["x"][::4], w["z"][::4], -3, color="#12355b", zorder=0)
    a1.axvspan(-G, G, color="gold", alpha=0.12)
    a1.axvspan(*XZ, ymin=0.0, ymax=1.0, fill=False, ec="0.7", ls="--", lw=1)
    a1.annotate("", xy=(-G * 0.3, 0.9), xytext=(-48, 0.9 + 48 * np.tan(np.radians(10)) * 0.25),
                arrowprops=dict(arrowstyle="-|>", color="gold", lw=2))
    a1.text(-49, 1.5, "radar beam →", color="gold", fontsize=10)
    a1.set(xlim=(w["x"][0], w["x"][-1]), ylim=(-2.2, 2.4))
    a1.set_title("whole 101 m solve window · gold = footprint ±g · line colour = induced HH current |J| (log) · "
                 "dashed = zoom below", color="0.85", fontsize=10)
    a1.set_ylabel("z [m]", color="0.8")
    # panel 2: |field| over the zoom, and a true-aspect close-up of the moving wavefronts
    a2.imshow(np.ma.masked_invalid(np.abs(P)), origin="lower", extent=(gx[0], gx[-1], gz[0], gz[-1]), aspect="auto",
              cmap="magma", vmin=0, vmax=2.4, interpolation="bilinear")
    a2.fill_between(gx, zs_on_grid, gz[0], color="#12355b")
    a2.plot(w["x"], w["z"], color="w", lw=0.8)
    a2.add_patch(plt.Rectangle((XC[0], zc - 0.35), XC[1] - XC[0], 1.2, fill=False, ec="cyan", lw=1.2))
    a2.set(xlim=XZ, ylim=(gz[0], gz[-1]))
    pos = a2.get_position()
    vex = ((XZ[1] - XZ[0]) / (pos.width * 16)) / ((gz[-1] - gz[0]) / (pos.height * 9))
    a2.set_title(f"|total HH field|: bright = incident + reflected interference, dark = shadow · vertical exaggeration ×{vex:.1f}",
                 color="0.85", fontsize=10)
    a2.set_ylabel("z [m]", color="0.8")
    phi = 2 * np.pi * f / 8
    a2c.imshow(np.ma.masked_invalid(np.real(PC * np.exp(1j * phi))), origin="lower",
               extent=(cx[0], cx[-1], zc + cz_rel[0], zc + cz_rel[-1]), aspect="equal", cmap="RdBu_r",
               vmin=-2.2, vmax=2.2, interpolation="bilinear")
    a2c.fill_between(cx, zs_c, zc + cz_rel[0], color="#12355b")
    a2c.plot(w["x"], w["z"], color="w", lw=1)
    a2c.set(xlim=XC, ylim=(zc + cz_rel[0], zc + cz_rel[-1]))
    a2c.set_title("cyan box, true aspect: wavefronts\n(radar phase slowed for display)", color="0.85", fontsize=10)
    # panel 3: time series
    a3.plot(ts, hh, color="#ff7b54", lw=1.6, label="HH")
    a3.plot(ts, vv, color="#59c3ff", lw=1.6, label="VV")
    a3.plot(ts[-1:], hh[-1:], "o", color="#ff7b54"); a3.plot(ts[-1:], vv[-1:], "o", color="#59c3ff")
    a3.axhline(-42.3, color="#ff7b54", ls=":", lw=1); a3.axhline(-15.3, color="#59c3ff", ls=":", lw=1)
    a3.set(xlim=(0, NF * DT), ylim=(-75, -5))
    a3.set_xlabel("sea time [s]", color="0.8"); a3.set_ylabel("σ⁰ backscatter [dB]", color="0.8")
    a3.set_title("backscatter from this patch vs sea time (dotted lines: Monte Carlo mean for this wind, angle and surface type)",
                 color="0.85", fontsize=10)
    a3.legend(loc="lower right", facecolor="#0b1020", labelcolor="0.9", fontsize=9)
    if cHH > 1e-4 or cVV > 1e-4:
        a3.text(0.01, 0.9, f"⚠ MOMI not converged this frame (HH {cHH:.0e}, VV {cVV:.0e})", color="red",
                transform=a3.transAxes)
    fig.savefig(png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"frame {f+1}/{NF}  t={t:.1f}s  HH {10*np.log10(sHH):6.1f} dB  VV {10*np.log10(sVV):6.1f} dB  "
          f"conv {cHH:.0e}/{cVV:.0e}  {time.time()-t0:.1f}s", flush=True)
print("done", flush=True)
