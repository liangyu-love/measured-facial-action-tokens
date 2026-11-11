#!/usr/bin/env bash
# Stronger text encoder on CMU-MOSI: deberta-v3-base (fp32 copy) for ALL methods, default recipe (6 ep, lr 2e-5, bs 16,
# best epoch by valid MAE), 5 seeds, 3 in parallel. Separate results dirs. Called by run_tune.sh after TUNE_A_DONE.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
export HF_HUB_OFFLINE=1 MSA_DATA=outputs/mosi MSA_DATASET=mosi ENCODER_DIR=$HOME/work/msa-pilot/models/deberta-v3-base-fp32
export RESULTS_DIR=results_deberta AUSOFT_RESULTS_DIR=results_deberta_ausoft AUSOFT2_RESULTS_DIR=results_deberta_acc PRE_RESULTS_DIR=results_deberta_pre
M=~/work/msa-pilot/outputs/mosi
ts() { echo "[$(date '+%F %T')] $*"; }
par() { local sc=$1 fl=$2 nm=$3
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "python $sc $fl $nm --seed {} > $M/logs/deberta_${nm}_s{}.log 2>&1 && echo \"[\$(date '+%F %T')] trained deberta $nm {}\" || echo \"[\$(date '+%F %T')] FAILED deberta $nm {}\""; }
ts "deberta MOSI start"
par train_bert.py --cond text
par train_bert.py --cond deva_av
par au_soft.py --variant mag
par au_soft.py --variant soft
par au_soft2.py --variant soft_emo
par pre_soft.py --variant au_soft_pre
par pre_soft.py --variant diff2_pre
ts "DEBERTA_MOSI_DONE"
