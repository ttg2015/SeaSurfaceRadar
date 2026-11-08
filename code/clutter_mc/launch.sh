#!/bin/bash
# usage: launch.sh PLAN NGPU NCPU [GPU_WEIGHT]
#   NGPU GPU shards + NCPU CPU shards; a GPU shard gets GPU_WEIGHT x as many seeds as a CPU shard.
#   Each shard writes results/<arm>_U<U>_dx20_s<i>.jsonl and resumes from it. Logs: results/logs.
cd "$(dirname "$0")"
mkdir -p results/logs
W=${4:-10}
N=$(( $2 + $3 )); NSLOTS=$(( $2 * W + $3 )); lo=0
for i in $(seq 0 $(( N - 1 ))); do
  if [ $i -lt $2 ]; then B=gpu; w=$W; else B=cpu; w=1; fi
  MC_BACKEND=$B MC_DX_DIV=20 MC_SHARD=$i MC_NSHARDS=$N MC_NSLOTS=$NSLOTS MC_SLOT_LO=$lo MC_SLOT_HI=$(( lo + w )) \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -W ignore run_mc.py $1 > results/logs/shard$i.log 2>&1 &
  lo=$(( lo + w ))
done
wait
