#!/usr/bin/env bash
# DIFF optimisation round on CH-SIMS v2 (runs alongside MOSEI; VRAM-aware: <=3 jobs, start a job only if >7 GB free).
# texp experts -> V1,V3..V7 -> V2,V8 (on V1 if V1 helps on valid) -> diff3_best -> analysis -> DIFF3_DONE
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
mkdir -p $F/logs
ts() { echo "[$(date '+%F %T')] $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
sched() {  # sched <jobfile>: each line = "<logname>|<command>"
  local pids=()
  while IFS= read -r line; do
    local lg=${line%%|*} cmd=${line#*|}
    while :; do
      pids=($(for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && echo $p; done))
      [ ${#pids[@]} -lt 3 ] && [ "$(free_mb)" -gt 7000 ] && break
      sleep 15
    done
    ( eval "$cmd" > $F/logs/$lg.log 2>&1 || echo "[$(date '+%F %T')] FAILED $lg" ) &
    pids+=($!); sleep 40  # let it allocate before the next VRAM check
  done < "$1"
  wait
}
J=$(mktemp)
ts "start"
for s in 1 2 3 4 5; do echo "texp_s$s|python texp.py --seed $s"; done > $J; sched $J; ts "text experts: $(ls $F/results_texp/*.pt 2>/dev/null | wc -l)"
: > $J
for s in 1 2 3 4 5; do
  for v in "diff3_texp|--texp" "diff3_frame|--frame" "diff3_head|--headcat" "diff3_rw05|--rw 0.5" "diff3_rw1|--rw 1" "diff3_ft|--ft" "diff3_moe|--moe"; do
    n=${v%%|*}; fl=${v#*|}; echo "${n}_s$s|python diff3.py --name $n --seed $s $fl" >> $J
  done
done
sched $J; ts "wave A done"
B=$(python diff3_analysis.py --decide-base 2>/dev/null); ts "V2/V8 base flags: [$B]"
: > $J
for s in 1 2 3 4 5; do echo "diff3_unc_s$s|python diff3.py --name diff3_unc --seed $s --unc $B" >> $J; echo "diff3_conf_s$s|python diff3.py --name diff3_conf --seed $s --conf $B" >> $J; done
sched $J; ts "wave B done"
ts "best: $(python diff3_analysis.py --decide-best 2>/dev/null)"
if [ "$(cat $F/results_diff3/best_flags.json)" != "{}" ]; then
  : > $J; for s in 1 2 3 4 5; do echo "diff3_best_s$s|python diff3.py --name diff3_best --seed $s --flags_file $F/results_diff3/best_flags.json" >> $J; done
  sched $J; ts "diff3_best done"
else ts "no variant beat diff2_pre on valid; diff3_best = diff2_pre"; fi
python diff3_analysis.py > $F/logs/diff3_analysis.log 2>&1
grep -h FAILED $F/run_diff3.log > /dev/null && ts "note: some runs FAILED (see above)"
ts "DIFF3_DONE"
