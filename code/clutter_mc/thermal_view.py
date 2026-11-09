"""Live window of the sweep's thermal log (results/thermal.csv, written by thermal_guard.py).
Reads the CSV only -- touches no sensors and nothing in the sweep. Refreshes every 30 s.

usage:  python thermal_view.py [hours]        # default: last 6 hours; 0 = everything
        python thermal_view.py [hours] --png  # just redraw figures/thermal_view.png and exit
"""
import csv, datetime, os, sys
import matplotlib
if "--png" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "thermal.csv")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
HOURS = float(args[0]) if args else 6.0

fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
try:
    fig.canvas.manager.set_window_title("Sweep thermals")
except Exception:
    pass


def load():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    t = [datetime.datetime.fromisoformat(r["time"]) for r in rows]
    if HOURS > 0 and t:
        t0 = t[-1] - datetime.timedelta(hours=HOURS)
        keep = [i for i, tt in enumerate(t) if tt >= t0]
        rows, t = [rows[i] for i in keep], [t[i] for i in keep]
    col = lambda k: [float(r[k]) if r[k] not in ("", "nan") else float("nan") for r in rows]
    return rows, t, col


def draw(_=None):
    try:
        rows, t, g = load()
    except Exception as e:
        ax[0].set_title(f"cannot read {CSV}: {e}")
        return
    if len(rows) < 2:
        return
    for a in ax:
        a.clear(); a.grid(alpha=0.3)
    last = rows[-1]
    ax[0].plot(t, g("gpu_C"), "C3", label="GPU")
    ax[0].plot(t, g("cpu_zone_C"), "C1", label="CPU / board zone")
    ax[0].axhline(83, color="C3", ls="--", lw=1, label="GPU pause 83")
    ax[0].axhline(88, color="C3", ls=":", lw=1, label="GPU stop 88")
    ax[0].axhline(85, color="C1", ls="--", lw=1, label="zone pause 85")
    ax[0].set_ylabel("°C"); ax[0].set_ylim(20, 95); ax[0].legend(fontsize=8, ncol=5, loc="lower left")
    ax[0].set_title(f"Sweep thermals — last sample {last['time']}:  GPU {last['gpu_C']} °C "
                    f"(margin {last['gpu_margin_C']}), zone {last['cpu_zone_C']} °C, "
                    f"GPU {last['gpu_W']} W, fan {last['gpu_fan']} %"
                    + (f"   [{last['action']}]" if last["action"] else ""), fontsize=10)
    ax[1].plot(t, g("gpu_margin_C"), "C3", label="GPU margin to its own slowdown limit")
    ax[1].axhline(5, color="C3", ls="--", lw=1, label="pause ≤ 5")
    ax[1].set_ylabel("°C"); ax[1].set_ylim(bottom=0); ax[1].legend(fontsize=8, loc="lower left")
    ax[2].plot(t, g("gpu_util"), "C3", lw=0.9, label="GPU load %")
    ax[2].plot(t, g("cpu_load_pct"), "C2", lw=0.9, label="CPU load %")
    ax[2].plot(t, g("cpu_perf_pct"), "C0", lw=0.9, label="CPU clock vs nominal %")
    ax[2].axhline(60, color="C0", ls="--", lw=1, label="throttle alarm < 60")
    ax[2].set_ylabel("%"); ax[2].set_ylim(0, max(220, max(g("cpu_perf_pct")) + 10))
    ax[2].legend(fontsize=8, ncol=4, loc="lower left")
    ax[3].plot(t, g("gpu_W"), "C4", lw=0.9, label="GPU power W")
    ax[3].plot(t, g("gpu_fan"), "C7", lw=0.9, label="GPU fan %")
    ax[3].set_ylim(bottom=0); ax[3].legend(fontsize=8, loc="lower left")
    for r, tt in zip(rows, t):
        if r["action"]:
            for a in ax:
                a.axvline(tt, color="k" if "STOP" in r["action"] else "0.5", lw=1)
            ax[0].text(tt, 94, " " + r["action"], fontsize=7, va="top", rotation=90)
    fig.autofmt_xdate(); fig.tight_layout()


if "--png" in sys.argv:
    draw()
    out = os.path.join(HERE, "figures", "thermal_view.png")
    fig.savefig(out, dpi=100)
    print(out)
else:
    anim = FuncAnimation(fig, draw, interval=30_000, cache_frame_data=False)
    draw()
    plt.show()
