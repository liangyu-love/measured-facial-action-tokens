#!/usr/bin/env bash
# CMU-MOSI training: bert-base-uncased, standard MMSA split, 5 seeds, 2 seeds in parallel (same seeds/configs).
# Keeps the Qwen3-VL queue paused (the densification queue resumes it).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATA=outputs/mosi MSA_DATASET=mosi ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-uncased
cd ~/work/msa-pilot/scripts
M=~/work/msa-pilot/outputs/mosi
ts() { echo "[$(date '+%F %T')] $*"; }
par() {  # par <script> <flag> <name> : seeds 1..5, 2 at a time
  local sc=$1 fl=$2 nm=$3
  printf "%s\n" 1 2 3 4 5 | xargs -P 2 -I{} bash -c "python $sc $fl $nm --seed {} > $M/logs/${nm}_s{}.log 2>&1 && echo \"[\$(date '+%F %T')] trained $nm {}\" || echo \"[\$(date '+%F %T')] FAILED $nm {}\""
}
until grep -q MOSI_PREP_DONE $M/run_prep.log 2>/dev/null; do sleep 30; done
until grep -q DIFF2_ALL_DONE ~/work/msa-pilot/outputs/full/run_diff2.log 2>/dev/null; do sleep 30; done
sleep 10; pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 paused"
par train_bert.py --cond text
par train_bert.py --cond deva_av
par au_soft.py --variant mag
par au_soft.py --variant soft
par au_soft2.py --variant soft_emo
python mosi_summary.py > $M/logs/summary.log 2>&1
ts "MOSI_ALL_DONE"
