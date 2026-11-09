#!/usr/bin/env bash
# Queued GPU jobs that must not share the GPU with the sweep (they fight over the 8 GB and both slow 2-4x).
# Waits for the sweep to finish, then runs the scattered-field animation data pass + render under its own
# thermal watchdog. Obeys results/PAUSE and results/STOP like the sweep (./sweep.sh pause|resume|stop).
cd "$(dirname "$0")"
export PATH="/usr/bin:/mingw64/bin:$PATH"
L=results/logs
until grep -q finished $L/launch.out 2>/dev/null; do sleep 60; done
[ -f results/STOP ] && { echo "STOP present; not starting $(date)"; exit 0; }
rm -f $L/after_sweep.done
TG_DONE=$L/after_sweep.done nohup python -W ignore thermal_guard.py >> $L/thermal_guard.log 2>&1 &
echo "animation data pass start $(date)"
python -W ignore scatter_anim.py 480 >> $L/scatter_anim.log 2>&1
echo "render start $(date)"
python -W ignore scatter_render.py both > $L/scatter_render.log 2>&1
echo "finished $(date)" > $L/after_sweep.done
echo "all done $(date)"
