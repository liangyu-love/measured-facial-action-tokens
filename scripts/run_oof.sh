#!/usr/bin/env bash
# Stacking-leakage fix: cross-fitted expert readings (oof_experts.py), then scalar-only DIFF with in-sample (diff3_sc, control)
# vs out-of-fold (diff3_oof) experts, 5 seeds each; audit refresh. VRAM-aware (<=2 jobs, start only if >8 GB free).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
sched() { local pids=()
  while IFS= read -r line; do local lg=${line%%|*} cmd=${line#*|}
    while :; do pids=($(for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && echo $p; done)); [ ${#pids[@]} -lt 2 ] && [ "$(free_mb)" -gt 8000 ] && break; sleep 20; done
    ( eval "$cmd" > $F/logs/$lg.log 2>&1 || echo "[$(date '+%F %T')] FAILED $lg" ) & pids+=($!); sleep 40
  done < "$1"; wait; }
J=$(mktemp)
ts "start"
: > $J; for s in 1 2 3 4 5; do echo "oof_experts_s$s|python oof_experts.py --seed $s" >> $J; done
sched $J; ts "OOF experts: $(ls $F/expert_oof/*.csv 2>/dev/null | wc -l); $(grep -h 'TRAIN Corr' $F/logs/oof_experts_s*.log | tr '\n' ' ')"
: > $J
for s in 1 2 3 4 5; do
  echo "dsc_s$s|DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_sc --precomp in --seed $s" >> $J
  echo "doof_s$s|DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_oof --precomp oof --seed $s" >> $J
done
sched $J; ts "diff3_sc / diff3_oof done"
python audit.py > $F/logs/audit_oof.log 2>&1; ts "OOF_DONE"
