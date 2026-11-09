#!/bin/bash
# Control the tail sweep (10,000 seas per wind). Safe to stop and restart at any time:
# every finished sea is saved immediately, and a restart skips everything already done.
#
#   ./sweep.sh start    launch 3 GPU + 6 CPU shards in the background (resumes where it left off)
#   ./sweep.sh stop     each shard finishes its current sea and exits; frees GPU/RAM (GPU ~5 s, CPU <~80 s)
#   ./sweep.sh pause    shards hold after their current sea (keeps a little GPU memory); instant resume
#   ./sweep.sh resume   continue after pause
#   ./sweep.sh status   progress per wind and per shard, last temperature sample
#
# A thermal watchdog (thermal_guard.py) runs alongside: it logs temperatures to results/thermal.csv,
# draws figures/thermal.png, pauses the sweep if the GPU or CPU runs hot and resumes it once cool,
# and stops it outright at hard limits (see the thresholds at the top of thermal_guard.py).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd "$(dirname "$0")"
R=results
case "$1" in
  start)
    rm -f $R/STOP $R/PAUSE
    if [ "$(grep -L -E '\] (done|STOP)' $R/logs/shard*.log 2>/dev/null | wc -l)" -gt 0 ] && \
       [ -n "$(find $R/logs -name 'shard*.log' -mmin -3 2>/dev/null)" ]; then
      echo "shards look like they are still running (logs updated <3 min ago); use ./sweep.sh status"; exit 1
    fi
    ( ./launch.sh big 3 6 14; echo "finished $(date)" ) > $R/logs/launch.out 2>&1 &
    sleep 2
    nohup python -W ignore thermal_guard.py >> $R/logs/thermal_guard.log 2>&1 &
    echo "sweep + thermal watchdog started (logs in $R/logs/, temperature graph: figures/thermal.png)";;
  stop)   touch $R/STOP;  echo "stop requested: shards exit after their current sea (GPU ~5 s, CPU up to ~80 s)";;
  pause)  touch $R/PAUSE; echo "pause requested";;
  resume) rm -f $R/PAUSE; echo "resumed";;
  status)
    [ -f $R/STOP ] && echo "STOP file present"; [ -f $R/PAUSE ] && echo "PAUSE file present"
    python - <<'PY'
import json, glob, os
from collections import defaultdict
seeds = defaultdict(set)
for p in glob.glob("results/full_U*_dx20_s*.jsonl"):
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        seeds[r["U"]].add(r["seed"])
for U in sorted(seeds): print(f"U = {U:4g} m/s: {len(seeds[U]):6d} / 10000 seas")
PY
    for f in $R/logs/shard*.log; do echo "  $(basename $f): $(tail -1 $f)"; done
    echo "  thermal (last sample): $(tail -1 $R/thermal.csv 2>/dev/null)";;
  *) sed -n '2,10p' "$0";;
esac
