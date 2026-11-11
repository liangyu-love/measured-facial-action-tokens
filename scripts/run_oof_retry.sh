#!/usr/bin/env bash
# Retry OOM-failed OOF expert seeds and their diff3_sc / diff3_oof runs (scripts skip finished outputs); final audit refresh.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
free_ok() { [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 9000 ]; }
until grep -q OOF_DONE $F/run_oof.log 2>/dev/null; do sleep 60; done
for round in 1 2 3; do
  for s in 1 2 3 4 5; do
    if [ ! -f $F/expert_oof/seed$s.csv ]; then until free_ok; do sleep 30; done; python oof_experts.py --seed $s > $F/logs/oof_experts_s${s}_r$round.log 2>&1 && ts "oof experts seed $s ok"; fi
    if [ -f $F/expert_oof/seed$s.csv ]; then
      for v in "diff3_sc in" "diff3_oof oof"; do set -- $v
        if [ ! -f $F/results_oof/$1_seed$s.json ]; then until free_ok; do sleep 30; done
          DIFF3_RESULTS_DIR=results_oof python diff3.py --name $1 --precomp $2 --seed $s > $F/logs/$1_s${s}_r$round.log 2>&1 && ts "$1 seed $s ok"; fi
      done
    fi
  done
done
python audit.py > $F/logs/audit_oof.log 2>&1; ts "OOF_RETRY_DONE"
