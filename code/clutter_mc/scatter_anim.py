"""Data pass for the scattered-field animations (rendered by scatter_render.py).

Same sea, beam and time axis as animate.py (seed 0, U = 10 m/s, choppy, theta = 80 deg, waves toward
the radar, 480 frames x 0.025 s), but it stores, per frame, the SCATTERED field psi_s only (no incident
beam) for HH and VV on a wide grid that includes the forward (specular) lobe, plus the induced currents,
the bistatic sigma0(theta_s) for both polarizations, and the complex backscatter amplitude (for Doppler).

Runs at lambda/20 so it fits on the GPU next to the sweep (lambda/40 needs ~5 GB per matrix). Every
frame's backscatter is compared with the lambda/40 value animate.py stored in results/animation.jsonl.

Scattered field from the solved current J (G = H0^(2)/(4j), e^{+j w t}):
  TE (HH, Dirichlet, J = dpsi/dn):  psi_s(r) = +(j/4) sum_n H0^(2)(kR) J_n s_n dx          (sign checked in animate.py)
  TM (VV, Neumann,   J = psi on S): psi_s(r) = +(jk/4) sum_n H1^(2)(kR) (r - r_n).(-zx_n, 1)/R J_n dx
     (the far-field limit of this is exactly the TM amplitude in mfie2d.nrcs; sign is re-checked below
      by requiring psi_i + psi_s -> J just above the surface)

usage: python scatter_anim.py [n_frames] [dt]       output: results/anim2/frame_XXXX.npz (resumable)
"""
import os, sys, json, time
import numpy as np
import torch

import surfaces as S
import gpusolve as GS
import run_mc as R
from surfaces import m

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 480
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.025
SEED, U, TH, DXDIV = 0, 10.0, np.radians(80), 20
LAM, K, G = R.LAM, R.K, R.G
N = R.L_WIN_LAM * DXDIV
DX = LAM / DXDIV
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "anim2")
os.makedirs(OUT, exist_ok=True)
TS = np.radians(np.linspace(-90, 90, 361))

sea0 = S.Sea(U, SEED, LAM, R.L_GEN_LAM["full"] * LAM)
sea0_w = sea0.window(R.L_WIN_LAM * 40, LAM / 40, True, demean=False)   # same offset as animate.py
z_off = sea0_w["z"].mean()


def surface(t):
    w = sea0.at_time(t).window(N, DX, True, demean=False)
    w["z"] = w["z"] - z_off
    return w


def solve(w, pol):
    x, z, zx, zxx = w["x"], w["z"], w["zx"], w["zxx"]
    A = GS.build_A(x, z, zx, zxx, K, pol)
    Ji = m.kirchhoff_current(x, z, zx, K, TH, G, pol)
    Js = GS.momi(A, Ji, 4)
    del A
    torch.cuda.empty_cache()
    return Js[-1], float(np.linalg.norm(Js[-1] - Js[-2]) / np.linalg.norm(Js[-1]))


def scat(xp, zp, w, J, pol, chunk=300):
    x, z, zx = w["x"], w["z"], w["zx"]
    xs, zs = torch.as_tensor(x, device="cuda"), torch.as_tensor(z, device="cuda")
    if pol == "TE":
        wt = torch.as_tensor(J * np.sqrt(1 + zx**2) * DX * 0.25j, device="cuda")
    else:
        wt = torch.as_tensor(J * DX * 0.25j * K, device="cuda")
        nx = torch.as_tensor(-zx, device="cuda")
    out = np.empty(len(xp), complex)
    for i0 in range(0, len(xp), chunk):
        X = torch.as_tensor(xp[i0:i0 + chunk], device="cuda")[:, None] - xs[None, :]
        Z = torch.as_tensor(zp[i0:i0 + chunk], device="cuda")[:, None] - zs[None, :]
        Rr = torch.hypot(X, Z)
        kR = K * Rr
        if pol == "TE":
            H = torch.complex(torch.special.bessel_j0(kR), -torch.special.bessel_y0(kR))
        else:
            H = torch.complex(torch.special.bessel_j1(kR), -torch.special.bessel_y1(kR)) * ((X * nx[None, :] + Z) / Rr)
        out[i0:i0 + chunk] = (H @ wt).cpu().numpy()
        del X, Z, Rr, kR, H
    return out


def amp(w, J, pol, ts):
    """Complex far-field amplitude I(theta_s) exactly as in mfie2d.nrcs (keeps the phase -> Doppler)."""
    x, z, zx = w["x"], w["z"], w["zx"]
    ph = np.exp(1j * K * (x * np.sin(ts) + z * np.cos(ts)))
    if pol == "TE":
        return np.sum(ph * J * np.sqrt(1 + zx**2)) * DX
    return np.sum(1j * K * (np.cos(ts) - zx * np.sin(ts)) * ph * J) * DX


# ---------------------------------------------------------------- sign checks (frame 0)
w = surface(0.0)
sel = np.nonzero(np.abs(w["x"]) < 8)[0][::100]
nrm = np.hypot(-w["zx"][sel], 1.0)
SIGN = {}
for pol, d in (("TE", 0.002), ("TM", DX)):
    J, _ = solve(w, pol)
    xp = w["x"][sel] - d * w["zx"][sel] / nrm
    zp = w["z"][sel] + d / nrm
    pi = np.asarray(m.thorsos_field(xp, zp, K, TH, G)[0])
    ps = scat(xp, zp, w, J, pol)
    ref = np.zeros_like(pi) if pol == "TE" else J[sel]          # Dirichlet: psi -> 0 ; Neumann: psi -> J
    scale = np.abs(pi) if pol == "TE" else np.abs(J[sel])
    rp = np.median(np.abs(pi + ps - ref) / scale)
    rm = np.median(np.abs(pi - ps - ref) / scale)
    SIGN[pol] = 1.0 if rp < rm else -1.0
    print(f"{pol} boundary check {d*1000:.0f} mm above surface: +sign {rp:.3f}, -sign {rm:.3f}", flush=True)
    assert min(rp, rm) < 0.25, f"{pol} scattered-field formula failed the boundary check"
    sig_nrcs = m.nrcs(w["x"], w["z"], w["zx"], J, K, TH, G, -TH, pol)[0]
    print(f"   nrcs {10*np.log10(sig_nrcs):.2f} dB", flush=True)

# ---------------------------------------------------------------- fixed field grid
XZ, ZTOP, H = (-30.0, 45.0), 7.0, LAM / 4
zin = w["z"][(w["x"] > XZ[0]) & (w["x"] < XZ[1])]
gx = np.arange(XZ[0], XZ[1], H)
gz = np.arange(zin.min() - 0.5, ZTOP, H)
GX, GZ = np.meshgrid(gx, gz)
np.savez(os.path.join(OUT, "grid.npz"), gx=gx, gz=gz, sign_TE=SIGN["TE"], sign_TM=SIGN["TM"], ts=TS, dt=DT,
         theta=TH, g=G, lam=LAM, dxdiv=DXDIV)
print(f"grid {len(gx)} x {len(gz)} = {GX.size} points", flush=True)

ref40 = {}
p40 = os.path.join(HERE, "results", "animation.jsonl")
if os.path.exists(p40):
    for l in open(p40):
        r = json.loads(l)
        ref40[r["frame"]] = r

for f in range(NF):
    fn = os.path.join(OUT, f"frame_{f:04d}.npz")
    if os.path.exists(fn):
        continue
    while os.path.exists(os.path.join(HERE, "results", "PAUSE")):   # obey the sweep's pause (gaming / thermal)
        time.sleep(5)
    if os.path.exists(os.path.join(HERE, "results", "STOP")):
        print("STOP file: exiting", flush=True); break
    t0 = time.time()
    t = f * DT
    w = surface(t)
    zsurf = np.interp(gx, w["x"], w["z"])
    above = GZ > zsurf[None, :] + 0.015            # keep >= ~1.3 dx off the surface (quadrature)
    rec = dict(t=t, x=w["x"][::4].astype(np.float32), z=w["z"][::4].astype(np.float32),
               zx=w["zx"][::4].astype(np.float32), jac_min=w["jac_min_window"])
    for pol, tag in (("TE", "HH"), ("TM", "VV")):
        J, conv = solve(w, pol)
        P = np.full(GX.shape, np.nan + 0j, np.complex64)
        P[above] = SIGN[pol] * scat(GX[above], GZ[above], w, J, pol)
        sig = m.nrcs(w["x"], w["z"], w["zx"], J, K, TH, G, TS, pol)
        rec.update({f"ps_{tag}": P, f"J_{tag}": np.abs(J[::4]).astype(np.float32), f"conv_{tag}": conv,
                    f"bis_{tag}": sig.astype(np.float32), f"I_{tag}": amp(w, J, pol, -TH),
                    f"sigma_{tag}": float(m.nrcs(w["x"], w["z"], w["zx"], J, K, TH, G, -TH, pol)[0])})
    if f in ref40:
        rec.update(sigma40_HH=ref40[f]["sigma_HH"], sigma40_VV=ref40[f]["sigma_VV"])
    np.savez_compressed(fn, **rec)
    d40 = (f"  vs lambda/40: HH {10*np.log10(rec['sigma_HH']/ref40[f]['sigma_HH']):+.2f} dB, "
           f"VV {10*np.log10(rec['sigma_VV']/ref40[f]['sigma_VV']):+.2f} dB") if f in ref40 else ""
    print(f"frame {f+1}/{NF} t={t:.2f}s HH {10*np.log10(rec['sigma_HH']):6.1f} VV {10*np.log10(rec['sigma_VV']):6.1f} dB "
          f"conv {rec['conv_HH']:.0e}/{rec['conv_VV']:.0e}{d40}  {time.time()-t0:.1f}s", flush=True)
print("done", flush=True)
