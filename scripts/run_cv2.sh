#!/usr/bin/env bash
# SIMS v2 5-fold CV, added baselines (2026-11-11): DEVA-style rule template (faithful 10 fps variant, k=8 as selected on the official
# validation split) and MAG-style gating. Same folds (make_cv.py), default recipe and 5 seeds as run_cv.sh (text, au_soft+emo).
# Two jobs in parallel when >7 GB VRAM is free. Note: DEVA prosody tertiles were fit on the official training split (label-free).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs; F=$R/full
ts() { echo "[$(date '+%F %T')] $*"; }
for k in 0 1 2 3 4; do
  for f in deva_text_dense_k8.jsonl audio_egemaps88.csv; do
    [ -e $R/full_cv/fold$k/$f ] || ln -s $F/$f $R/full_cv/fold$k/$f
  done
done
run() {  # fold seed name cmd...
  local k=$1 s=$2 n=$3; shift 3
  MSA_DATA=outputs/full_cv/fold$k nice -n 10 "$@" --seed $s > $R/full_cv/fold$k/logs/${n}_s$s.log 2>&1 && ts "fold$k $n $s" || ts "FAILED fold$k $n $s"
}
for k in 0 1 2 3 4; do for s in 1 2 3 4 5; do
  for n in deva_dense mag; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 7000 ]; do sleep 30; done
    if [ $n = deva_dense ]; then
      ( export DEVA_FILE=deva_text_dense_k8.jsonl RESULTS_DIR=results_deva_dense; run $k $s $n python train_bert.py --cond deva_av ) &
    else
      run $k $s $n python au_soft.py --variant mag &
    fi
    sleep 90  # let the job allocate memory before checking again
  done
done; done
wait
ts "CV2_DONE"
