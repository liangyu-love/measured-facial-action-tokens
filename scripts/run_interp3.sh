#!/usr/bin/env bash
# Train one diff3_unc (seed 1, same recipe) with the model saved, then the Conflict-clip interpretability analysis.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 8000 ]; do sleep 30; done
ts "train"
DIFF3_SAVE=$F/interp_diff3_unc_seed1.ckpt DIFF3_RESULTS_DIR=_interp_tmp python diff3.py --name diff3_unc --seed 1 --texp --unc > $F/logs/interp_train.log 2>&1 || { ts "TRAIN FAILED"; exit 1; }
ts "trained: $(python -c "import json;r=json.load(open('$F/_interp_tmp/diff3_unc_seed1.json'));print(r['test']['acc2'], r['test_conflict']['acc2'])")"
python interp3.py --ckpt $F/interp_diff3_unc_seed1.ckpt --seed 1 > $F/logs/interp3.log 2>&1 && ts "INTERP3_DONE" || ts "INTERP3 FAILED"
