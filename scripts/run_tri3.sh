#!/usr/bin/env bash
# Step 4: three cross-fitted experts (text, face, HuBERT voice) as scalar DIFF, with / without whom-to-trust head; 5 seeds each.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full DIFF3_RESULTS_DIR=results_oof
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
for s in 1 2 3 4 5; do
  for v in "diff3_tri3" "diff3_tri3_trust --trust 0.1"; do set -- $v; nm=$1; shift
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 8000 ]; do sleep 30; done
    python diff3.py --name $nm --precomp oof --audio_oof --seed $s "$@" > $F/logs/${nm}_s$s.log 2>&1 && ts "$nm $s" || ts "FAILED $nm $s"
  done
done
python audit.py > $F/logs/audit_tri3.log 2>&1; ts "TRI3_DONE"
