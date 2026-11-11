#!/usr/bin/env bash
# Capacity control: same as diff3_oof (cross-fitted readings) but the VISUAL reading zeroed / shuffled; text expert intact. 5 seeds.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
free_ok() { [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 8000 ]; }
until [ "$(ls $F/expert_oof/seed[1-5].csv 2>/dev/null | wc -l)" -ge 5 ]; do sleep 60; done
ts "start"
for s in 1 2 3 4 5; do for c in zero shuf; do
  until free_ok; do sleep 30; done
  DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_v$c --precomp oof --vctl $c --seed $s > $F/logs/capctl_${c}_s$s.log 2>&1 && ts "v$c $s" || ts "FAILED v$c $s"
done; done
until grep -q OOF_RETRY_DONE $F/run_oof_retry.log 2>/dev/null; do sleep 60; done
python audit.py > $F/logs/audit_capctl.log 2>&1; ts "CAPCTL_DONE"
