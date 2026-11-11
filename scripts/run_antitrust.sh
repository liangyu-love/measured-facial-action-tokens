#!/usr/bin/env bash
# Anti-over-trust variants (scalar DIFF): diff3_noise, diff3_drop, diff3_reg; 5 seeds; after the OOF expert readings exist.
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
: > $J; for s in 1 2 3 4 5; do echo "reg_experts_s$s|python reg_experts.py --seed $s" >> $J; done
sched $J; ts "reg experts: $(grep -h 'text(reg)' $F/logs/reg_experts_s1.log | tr '\n' ' ')"
until [ "$(ls $F/expert_oof/seed[1-5].csv 2>/dev/null | wc -l)" -ge 5 ]; do sleep 60; done
: > $J
for s in 1 2 3 4 5; do
  echo "dnoise_s$s|DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_noise --precomp in --noise --seed $s" >> $J
  echo "ddrop_s$s|DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_drop --precomp in --ddrop 0.3 --seed $s" >> $J
  echo "dreg_s$s|DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_reg --precomp reg --seed $s" >> $J
done
sched $J; ts "variants done"
until grep -q OOF_DONE $F/run_oof.log 2>/dev/null; do sleep 60; done
python audit.py > $F/logs/audit_antitrust.log 2>&1; ts "ANTITRUST_DONE"
