"""Thermal watchdog for the sweep. Samples every 15 s, logs to results/thermal.csv, redraws
figures/thermal.png every 5 min, and pauses / stops the sweep through the same control files
sweep.sh uses (results/PAUSE, results/STOP).

Sensors (no admin rights needed):
  GPU  nvidia-smi: temperature, margin to the GPU's own slowdown limit (temperature.gpu.tlimit),
       hardware/software thermal-slowdown flags, power, fan, utilisation
  CPU  ACPI thermal zone (motherboard-level, lags the cores) and "% Processor Performance"
       (clock vs nominal; collapses when the CPU thermally throttles under load). A true core
       temperature needs admin rights or a third-party tool, which this deliberately avoids.

Rules (sustained = 2 consecutive samples = 30 s):
  PAUSE  GPU >= 83 C, or GPU margin <= 5 C, or any GPU thermal slowdown, or CPU zone >= 85 C,
         or CPU performance < 60 % while CPU load > 80 % (throttling)
  STOP   GPU >= 88 C, or GPU margin <= 2 C, or CPU zone >= 95 C  (no automatic restart)
  RESUME only a pause this watchdog made, after 3 min with GPU <= 72 C, margin >= 15 C,
         CPU zone <= 75 C and no throttling. A manual pause (./sweep.sh pause) is never lifted.
"""
import csv, os, subprocess, time, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
CSV = os.path.join(R, "thermal.csv")
PNG = os.path.join(HERE, "figures", "thermal.png")
PAUSE, STOP = os.path.join(R, "PAUSE"), os.path.join(R, "STOP")
DONE = os.environ.get("TG_DONE", os.path.join(R, "logs", "launch.out"))   # exit when this says "finished"
G_PAUSE, G_STOP, M_PAUSE, M_STOP, G_SLOW = 83, 88, 5, 2, True
C_PAUSE, C_STOP, PERF_MIN = 85, 95, 60
G_OK, M_OK, C_OK, OK_FOR = 72, 15, 75, 180
FIELDS = ["time", "gpu_C", "gpu_margin_C", "gpu_slowdown", "gpu_W", "gpu_fan", "gpu_util", "cpu_zone_C",
          "cpu_perf_pct", "cpu_load_pct", "action"]


def gpu():
    q = ("temperature.gpu,temperature.gpu.tlimit,clocks_event_reasons.hw_thermal_slowdown,"
         "clocks_event_reasons.sw_thermal_slowdown,power.draw,fan.speed,utilization.gpu")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout.strip().split(", ")
    f = lambda s: float(s) if s.replace(".", "", 1).isdigit() else float("nan")
    return dict(gpu_C=f(out[0]), gpu_margin_C=f(out[1]),
                gpu_slowdown=int(out[2].strip() == "Active" or out[3].strip() == "Active"),
                gpu_W=f(out[4]), gpu_fan=f(out[5]), gpu_util=f(out[6]))


def cpu():
    ps = ("$z=(Get-Counter '\\Thermal Zone Information(*)\\Temperature').CounterSamples | Measure-Object CookedValue -Maximum;"
          "$p=(Get-Counter '\\Processor Information(_Total)\\% Processor Performance').CounterSamples[0].CookedValue;"
          "$l=(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue;"
          "\"$($z.Maximum) $p $l\"")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
    z, p, l = (float(v) for v in out.stdout.split())
    return dict(cpu_zone_C=z - 273.15, cpu_perf_pct=p, cpu_load_pct=l)


def plot():
    rows = list(csv.DictReader(open(CSV)))
    if len(rows) < 2:
        return
    t = [datetime.datetime.fromisoformat(r["time"]) for r in rows]
    g = lambda k: [float(r[k]) for r in rows]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(t, g("gpu_C"), "C3", label="GPU temperature")
    ax[0].plot(t, g("cpu_zone_C"), "C1", label="CPU / board thermal zone")
    ax[0].axhline(G_PAUSE, color="C3", ls="--", lw=1, label=f"GPU pause {G_PAUSE} °C")
    ax[0].axhline(G_STOP, color="C3", ls=":", lw=1, label=f"GPU stop {G_STOP} °C")
    ax[0].axhline(C_PAUSE, color="C1", ls="--", lw=1, label=f"zone pause {C_PAUSE} °C")
    ax[0].set_ylabel("°C"); ax[0].legend(fontsize=8, ncol=3); ax[0].set_title("Sweep thermal log")
    ax[1].plot(t, g("gpu_margin_C"), "C3", label="GPU margin to its own slowdown limit")
    ax[1].axhline(M_PAUSE, color="C3", ls="--", lw=1, label=f"pause ≤ {M_PAUSE} °C")
    ax[1].set_ylabel("°C"); ax[1].legend(fontsize=8)
    ax[2].plot(t, g("cpu_perf_pct"), "C0", label="CPU % performance (clock vs nominal)")
    ax[2].plot(t, g("cpu_load_pct"), "0.6", lw=0.8, label="CPU load %")
    ax[2].plot(t, g("gpu_util"), "C3", lw=0.8, alpha=0.6, label="GPU load %")
    ax[2].axhline(PERF_MIN, color="C0", ls="--", lw=1, label=f"throttle alarm < {PERF_MIN} %")
    ax[2].set_ylabel("%"); ax[2].legend(fontsize=8, ncol=2)
    for r, tt in zip(rows, t):
        if r["action"]:
            for a in ax:
                a.axvline(tt, color="k" if "STOP" in r["action"] else "0.5", lw=1)
            ax[0].text(tt, ax[0].get_ylim()[1], " " + r["action"], fontsize=7, va="top", rotation=90)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(PNG, dpi=110); plt.close(fig)


def main():
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    hot, ok_since, last_plot = 0, None, 0
    while True:
        try:
            s = {**gpu(), **cpu()}
        except Exception as e:  # a failed read never stops the sweep, but is logged
            print("sensor read failed:", e, flush=True)
            time.sleep(15)
            continue
        action = ""
        stop = s["gpu_C"] >= G_STOP or s["gpu_margin_C"] <= M_STOP or s["cpu_zone_C"] >= C_STOP
        pause = (s["gpu_C"] >= G_PAUSE or s["gpu_margin_C"] <= M_PAUSE or (G_SLOW and s["gpu_slowdown"])
                 or s["cpu_zone_C"] >= C_PAUSE or (s["cpu_perf_pct"] < PERF_MIN and s["cpu_load_pct"] > 80))
        hot = hot + 1 if (pause or stop) else 0
        if stop and hot >= 2 and not os.path.exists(STOP):
            open(STOP, "w").write("thermal"); action = "THERMAL STOP"
        elif pause and hot >= 2 and not os.path.exists(PAUSE):
            open(PAUSE, "w").write("thermal"); action = "thermal pause"
        cool = (s["gpu_C"] <= G_OK and s["gpu_margin_C"] >= M_OK and s["cpu_zone_C"] <= C_OK
                and not s["gpu_slowdown"] and not (s["cpu_perf_pct"] < PERF_MIN and s["cpu_load_pct"] > 80))
        ok_since = (ok_since or time.time()) if cool else None
        if (os.path.exists(PAUSE) and open(PAUSE).read().strip() == "thermal" and not os.path.exists(STOP)
                and ok_since and time.time() - ok_since >= OK_FOR):
            os.remove(PAUSE); action = "thermal resume"
        w.writerow({"time": datetime.datetime.now().isoformat(timespec="seconds"), **{k: round(v, 2) for k, v in s.items()},
                    "action": action}); f.flush()
        if action:
            print(datetime.datetime.now().strftime("%H:%M:%S"), action, s, flush=True)
        if time.time() - last_plot > 300 or action:
            try:
                plot()
            except Exception as e:
                print("plot failed:", e, flush=True)
            last_plot = time.time()
        if os.path.exists(DONE) and "finished" in open(DONE).read():
            plot(); print("sweep finished; watchdog exiting", flush=True); return
        time.sleep(15)


if __name__ == "__main__":
    main()
