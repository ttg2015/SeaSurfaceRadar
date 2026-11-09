"""Render the data from scatter_anim.py into two animations:

  figures/scattered_field.webp/.gif   just the scattered field |psi_s| (dB re incident beam peak), HH over VV
  figures/radar_dashboard.webp/.gif   everything at once: surface + currents, scattered fields, bistatic
                                      sigma0, Doppler spectrogram, backscatter time series, frame stats

usage: python scatter_render.py [which]   which in {field, dash, both} (default both)
Frames go to figures/anim2_field/ and figures/anim2_dash/ (gitignored); encoding with Pillow.
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "anim2")
which = sys.argv[1] if len(sys.argv) > 1 else "both"
gr = np.load(os.path.join(D, "grid.npz"))
gx, gz, TS, DT, TH, G = gr["gx"], gr["gz"], gr["ts"], float(gr["dt"]), float(gr["theta"]), float(gr["g"])
files = sorted(glob.glob(os.path.join(D, "frame_*.npz")))
NF = len(files)
print(f"{NF} frames", flush=True)
BG, FG = "#0b1020", "0.85"
HHc, VVc = "#ff7b54", "#59c3ff"

# ---------------------------------------------------------------- whole-record quantities (light: no fields)
sc = {k: [] for k in ("t", "sigma_HH", "sigma_VV", "I_HH", "I_VV", "bis_HH", "bis_VV", "conv_HH", "conv_VV",
                      "sigma40_HH", "sigma40_VV", "jac_min")}
for fn in files:
    with np.load(fn) as d:
        for k in sc:
            sc[k].append(d[k] if k in d.files else np.nan)
sc = {k: np.array(v) for k, v in sc.items()}
t = sc["t"]
hh, vv = 10 * np.log10(sc["sigma_HH"]), 10 * np.log10(sc["sigma_VV"])
bis_mean = {p: 10 * np.log10(np.mean(sc[f"bis_{p}"], axis=0)) for p in ("HH", "VV")}

# Doppler: sliding-window spectrogram of the complex backscatter amplitude (positive = approaching)
NW, NFFT = 80, 256
fd = np.fft.fftshift(np.fft.fftfreq(NFFT, DT))
win = np.hanning(NW)


def spec(I):
    out = np.full((NF, NFFT), np.nan)
    for i in range(NF):
        lo, hi = i - NW // 2, i + NW // 2
        if lo < 0 or hi > NF:
            continue
        s = np.fft.fftshift(np.fft.fft((I[lo:hi] - I[lo:hi].mean() * 0) * win, NFFT))
        out[i] = 10 * np.log10(np.abs(s) ** 2 + 1e-300)
    return out


SPEC = {p: spec(sc[f"I_{p}"]) for p in ("HH", "VV")}
full = {}
for p in ("HH", "VV"):
    w = np.hanning(NF)
    s = np.abs(np.fft.fftshift(np.fft.fft(sc[f"I_{p}"] * w, 4096))) ** 2
    full[p] = (np.fft.fftshift(np.fft.fftfreq(4096, DT)), 10 * np.log10(s / s.max()))
KB = 2 * (2 * np.pi / float(gr["lam"])) * np.sin(TH)
FB = np.sqrt(9.81 * KB + 7.4e-5 * KB**3) / (2 * np.pi)
peak = {p: full[p][0][np.argmax(full[p][1])] for p in ("HH", "VV")}
print(f"Bragg Doppler {FB:.2f} Hz; whole-record peaks HH {peak['HH']:+.2f} Hz, VV {peak['VV']:+.2f} Hz", flush=True)
d40 = {p: 10 * np.log10(sc[f"sigma_{p}"] / sc[f"sigma40_{p}"]) for p in ("HH", "VV")}
print("lambda/20 vs lambda/40 backscatter: " + ", ".join(
    f"{p} median |d| {np.nanmedian(np.abs(d40[p])):.2f} dB, max {np.nanmax(np.abs(d40[p])):.2f} dB" for p in d40), flush=True)


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="0.75", labelsize=8)
    for s in ax.spines.values():
        s.set_color("0.35")


def field_img(ax, d, tag, title, cbar=None):
    P = d[f"ps_{tag}"]
    dB = 20 * np.log10(np.abs(P) + 1e-12)
    im = ax.imshow(np.ma.masked_invalid(np.where(np.isnan(P.real), np.nan, dB)), origin="lower",
                   extent=(gx[0], gx[-1], gz[0], gz[-1]), aspect="auto", cmap="magma", vmin=-40, vmax=3,
                   interpolation="bilinear")
    x, z = d["x"], d["z"]
    sel = (x > gx[0] - 1) & (x < gx[-1] + 1)
    ax.fill_between(x[sel], z[sel], gz[0], color="#12355b")
    ax.plot(x[sel], z[sel], color="w", lw=0.7)
    ax.set(xlim=(gx[0], gx[-1]), ylim=(gz[0], gz[-1]))
    ax.set_title(title, color=FG, fontsize=10)
    ax.set_ylabel("z [m]", color="0.75", fontsize=9)
    return im


def save(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- 1. scattered field only
def render_field(i, d, out):
    fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[60, 1], hspace=0.22, wspace=0.02, left=0.05, right=0.95,
                          top=0.9, bottom=0.07)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for ax in (a, b, c):
        style(ax)
    fig.suptitle(f"Scattered field only (incident beam subtracted) · 10° grazing, choppy 10 m/s sea · "
                 f"sea time {d['t']:5.2f} s", color="w", fontsize=14)
    im = field_img(a, d, "HH", f"HH   |ψ_s| in dB re incident beam peak · backscatter σ⁰ = {10*np.log10(d['sigma_HH']):.1f} dB")
    field_img(b, d, "VV", f"VV   |ψ_s| · backscatter σ⁰ = {10*np.log10(d['sigma_VV']):.1f} dB")
    b.set_xlabel("x [m]   (radar far off to the left; vertical exaggeration ≈ ×2)", color="0.75", fontsize=9)
    for ax in (a, b):
        ax.axvspan(-G, G, ymax=0.03, color="gold", alpha=0.6)
        ax.annotate("forward / specular reflection", xy=(38, 5.6), xytext=(18, 6.2), color="0.9", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="0.9"))
        ax.text(-29, 6.2, "← back toward the radar (this is the clutter)", color="0.9", fontsize=9)
    cb = fig.colorbar(im, cax=c)
    cb.set_label("dB", color="0.75"); cb.ax.tick_params(colors="0.75", labelsize=8)
    save(fig, out)


# ---------------------------------------------------------------- 2. dashboard
def render_dash(i, d, out):
    fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=BG)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.75, 1.1, 1.1, 0.8], width_ratios=[1.6, 1.0, 0.9],
                          hspace=0.55, wspace=0.22, left=0.045, right=0.985, top=0.92, bottom=0.06)
    A0 = fig.add_subplot(gs[0, :2]); TX = fig.add_subplot(gs[0, 2])
    F1 = fig.add_subplot(gs[1, :2]); BI = fig.add_subplot(gs[1, 2])
    F2 = fig.add_subplot(gs[2, :2]); SP = fig.add_subplot(gs[2, 2])
    TSr = fig.add_subplot(gs[3, :2]); FS = fig.add_subplot(gs[3, 2])
    for ax in (A0, TX, F1, BI, F2, SP, TSr, FS):
        style(ax)
    fig.suptitle(f"Radar at 10° grazing on a choppy 10 m/s sea moving toward the radar — everything at once · "
                 f"sea time {d['t']:5.2f} s", color="w", fontsize=14)
    # surface + currents
    x, z = d["x"], d["z"]
    A0.fill_between(x, z, -3, color="#12355b")
    A0.plot(x, z, color="w", lw=0.8)
    A0.axvspan(-G, G, color="gold", alpha=0.1)
    A0.set(xlim=(x[0], x[-1]), ylim=(-2.2, 2.2)); A0.set_ylabel("z [m]", color="0.75", fontsize=9)
    a2 = A0.twinx(); style(a2)
    a2.semilogy(x, d["J_HH"] / d["J_HH"].max(), color=HHc, lw=0.8, label="|J| HH")
    a2.semilogy(x, d["J_VV"] / d["J_VV"].max(), color=VVc, lw=0.8, label="|J| VV")
    a2.set_ylim(1e-4, 3); a2.legend(loc="upper left", fontsize=8, facecolor=BG, labelcolor="0.9", ncol=2)
    A0.set_title("101 m solve window: surface (white), footprint ±g (gold), induced surface currents (normalised, log)",
                 color=FG, fontsize=9)
    # scattered fields
    field_img(F1, d, "HH", "HH scattered field |ψ_s| [dB re incident peak, −40…+3]")
    field_img(F2, d, "VV", "VV scattered field |ψ_s|")
    for ax in (F1, F2):
        ax.axvspan(-G, G, ymax=0.03, color="gold", alpha=0.6)
    # stats
    TX.axis("off")
    zx = d["zx"]
    inwin = np.abs(x) < G
    lines = [f"frame {i+1}/{NF}    t = {d['t']:.2f} s",
             f"σ⁰ HH {10*np.log10(d['sigma_HH']):6.1f} dB   VV {10*np.log10(d['sigma_VV']):6.1f} dB",
             f"HH − VV  {10*np.log10(d['sigma_HH']/d['sigma_VV']):6.1f} dB",
             f"λ/20 vs λ/40 check: HH {d40['HH'][i]:+.2f}, VV {d40['VV'][i]:+.2f} dB",
             f"MOMI residual  {float(d['conv_HH']):.0e} / {float(d['conv_VV']):.0e}",
             f"footprint max slope {np.degrees(np.arctan(np.abs(zx[inwin]).max())):.0f}°,  min dx/du {float(d['jac_min']):.2f}",
             f"(dx/du → 0 would be an overturning crest)"]
    TX.text(0.0, 1.0, "\n".join(lines), color="0.9", fontsize=9.5, family="monospace", va="top", transform=TX.transAxes)
    # bistatic
    deg = np.degrees(TS)
    for p, c in (("HH", HHc), ("VV", VVc)):
        BI.plot(deg, bis_mean[p], color=c, lw=0.8, ls=":", alpha=0.8)
        BI.plot(deg, 10 * np.log10(d[f"bis_{p}"]), color=c, lw=1.2, label=p)
    BI.axvline(-80, color="0.6", lw=0.8, ls="--"); BI.text(-79, 22, "back to\nradar", color="0.7", fontsize=7, va="top")
    BI.axvline(80, color="0.6", lw=0.8, ls="--"); BI.text(79, 22, "specular", color="0.7", fontsize=7, va="top", ha="right")
    BI.set(xlim=(-90, 90), ylim=(-70, 25)); BI.set_xlabel("scattering angle θ_s [deg]", color="0.75", fontsize=8)
    BI.set_title("bistatic σ⁰ (solid now, dotted 12 s mean)", color=FG, fontsize=9)
    BI.legend(fontsize=7, facecolor=BG, labelcolor="0.9", loc="lower center", ncol=2)
    # spectrogram (HH), revealed up to now
    S = SPEC["HH"].copy(); S[i + 1:] = np.nan
    ref = np.nanmax(SPEC["HH"])
    SP.imshow((S - ref).T, origin="lower", aspect="auto", extent=(t[0], t[-1] + DT, fd[0], fd[-1]), cmap="viridis",
              vmin=-40, vmax=0, interpolation="nearest")
    SP.axhline(FB, color="w", ls=":", lw=0.8); SP.axhline(-FB, color="w", ls=":", lw=0.8)
    SP.text(t[-1], FB + 0.4, "Bragg ±", color="w", fontsize=7, ha="right")
    SP.set(ylim=(-12, 12), xlim=(t[0], t[-1])); SP.set_ylabel("Doppler [Hz]", color="0.75", fontsize=8)
    SP.set_title("HH Doppler spectrogram (2 s window, + = approaching)", color=FG, fontsize=9)
    # time series
    TSr.plot(t[:i + 1], hh[:i + 1], color=HHc, lw=1.3, label="HH")
    TSr.plot(t[:i + 1], vv[:i + 1], color=VVc, lw=1.3, label="VV")
    TSr.plot(t[i], hh[i], "o", color=HHc); TSr.plot(t[i], vv[i], "o", color=VVc)
    TSr.axhline(-42.3, color=HHc, ls=":", lw=1); TSr.axhline(-15.3, color=VVc, ls=":", lw=1)
    TSr.set(xlim=(0, t[-1]), ylim=(-75, -5)); TSr.set_xlabel("sea time [s]", color="0.75", fontsize=8)
    TSr.set_ylabel("σ⁰ [dB]", color="0.75", fontsize=8)
    TSr.set_title("backscatter vs time (dotted: Monte Carlo mean, same wind/angle/surface)", color=FG, fontsize=9)
    TSr.legend(fontsize=7, facecolor=BG, labelcolor="0.9", loc="lower right", ncol=2)
    # whole-record Doppler spectra
    for p, c in (("HH", HHc), ("VV", VVc)):
        FS.plot(full[p][0], full[p][1], color=c, lw=1, label=p)
    FS.axvline(FB, color="0.6", ls=":", lw=0.8); FS.axvline(-FB, color="0.6", ls=":", lw=0.8)
    FS.set(xlim=(-12, 12), ylim=(-50, 2)); FS.set_xlabel("Doppler [Hz]", color="0.75", fontsize=8)
    FS.set_title("Doppler spectrum, whole 12 s", color=FG, fontsize=9)
    FS.legend(fontsize=7, facecolor=BG, labelcolor="0.9", loc="upper left")
    save(fig, out)


def encode(frame_dir, stem, fps=20, gw=640):
    fs = sorted(glob.glob(os.path.join(frame_dir, "f_*.png")))
    frames = [Image.open(f).convert("RGB") for f in fs]
    dur = int(round(1000 / fps))
    webp = os.path.join(HERE, "figures", stem + ".webp")
    frames[0].save(webp, save_all=True, append_images=frames[1:], duration=dur, loop=0, quality=70, method=4)
    small = [im.resize((gw, round(im.height * gw / im.width)), Image.LANCZOS) for im in frames]
    pal = small[len(small) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gq = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in small]
    gif = os.path.join(HERE, "figures", stem + ".gif")
    gq[0].save(gif, save_all=True, append_images=gq[1:], duration=dur, loop=0)
    for p in (webp, gif):
        print(f"{os.path.basename(p)}: {len(fs)} frames, {os.path.getsize(p)/1e6:.1f} MB", flush=True)


jobs = [(k, fn, sub, stem) for k, fn, sub, stem in
        (("field", render_field, "anim2_field", "scattered_field"), ("dash", render_dash, "anim2_dash", "radar_dashboard"))
        if which in (k, "both")]
for _, fn, sub, _ in jobs:
    os.makedirs(os.path.join(HERE, "figures", sub), exist_ok=True)
for i, path in enumerate(files):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    for _, fn, sub, _ in jobs:
        fn(i, d, os.path.join(HERE, "figures", sub, f"f_{i:04d}.png"))
    if (i + 1) % 40 == 0:
        print(f"rendered {i+1}/{NF}", flush=True)
for _, _, sub, stem in jobs:
    encode(os.path.join(HERE, "figures", sub), stem)
