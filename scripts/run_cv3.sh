#!/usr/bin/env bash
# SIMS v2 5-fold CV under the TUNED configurations (2026-11-11, reviewer-style check): same folds (make_cv.py) and 5 seeds as
# run_cv.sh / run_cv2.sh, but every method uses the configuration selected on the OFFICIAL validation split (outputs/full/tuned_configs.json;
# no re-tuning per fold, so no extra selection on CV data). Early stopping (patience 3) on each fold's own valid split, as in tune.py stage 2.
# Results go to outputs/full_cv/fold<k>/cvt/<method>/ (default-recipe CV results are left untouched). 3 jobs in parallel.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-chinese
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs
ts() { echo "[$(date '+%F %T')] $*"; }
job() {  # fold seed method
  local k=$1 s=$2 m=$3 d=$R/full_cv/fold$1
  local log=$d/logs/cvt_${m}_s$s.log
  ls $d/cvt/$m/*_seed${s}_test_pred.csv > /dev/null 2>&1 && { ts "skip fold$k $m $s"; return; }
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 5000 ]; do sleep 30; done
  export MSA_DATA=outputs/full_cv/fold$k
  case $m in
    text)        RESULTS_DIR=cvt/text nice -n 10 python train_bert.py --cond text --seed $s --epochs 15 --lr 2e-5 --lr_new 1e-3 --loss l1 --patience 3 > $log 2>&1 ;;
    deva_dense)  DEVA_FILE=deva_text_dense_k8.jsonl RESULTS_DIR=cvt/deva_dense nice -n 10 python train_bert.py --cond deva_av --seed $s --epochs 10 --lr 3e-5 --lr_new 1e-4 --loss mse --patience 3 > $log 2>&1 ;;
    mag)         AUSOFT_RESULTS_DIR=cvt/mag nice -n 10 python au_soft.py --variant mag --seed $s --epochs 10 --lr 2e-5 --lr_new 1e-3 --loss mse --patience 3 > $log 2>&1 ;;
    au_soft_emo) AUSOFT2_RESULTS_DIR=cvt/au_soft_emo nice -n 10 python au_soft2.py --variant soft_emo --seed $s --epochs 10 --lr 3e-5 --lr_new 1e-3 --loss l1 --patience 3 > $log 2>&1 ;;
  esac && ts "fold$k $m $s" || ts "FAILED fold$k $m $s"
}
export -f job ts; export R
if [ "$1" = smoke ]; then job 0 1 mag; job 0 1 au_soft_emo; ts "CV3_SMOKE_DONE"; exit; fi
for k in 0 1 2 3 4; do for s in 1 2 3 4 5; do for m in text deva_dense mag au_soft_emo; do echo "$k $s $m"; done; done; done \
  | xargs -P 3 -L 1 bash -c 'sleep $((RANDOM % 20)); job $0 $1 $2'
python cv_compare_tuned.py > $R/full/logs/cv_compare_tuned.log 2>&1
ts "CV3_DONE"
