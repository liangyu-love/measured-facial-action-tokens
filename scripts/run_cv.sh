#!/usr/bin/env bash
# Low-priority SIMS v2 5-fold CV (text, au_soft+emo; default recipe; 5 seeds). One job at a time, only when >10 GB VRAM is free.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs
ts() { echo "[$(date '+%F %T')] $*"; }
python make_cv.py
for k in 0 1 2 3 4; do for s in 1 2 3 4 5; do
  for job in "train_bert.py --cond text" "au_soft2.py --variant soft_emo"; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 10000 ]; do sleep 60; done
    MSA_DATA=outputs/full_cv/fold$k nice -n 10 python $job --seed $s > $R/full_cv/fold$k/logs/$(echo $job | awk '{print $NF}')_s$s.log 2>&1 && ts "fold$k $job $s" || ts "FAILED fold$k $job $s"
  done
done; done
python audit.py > $R/full/logs/audit_cv.log 2>&1; ts "CV_DONE"
