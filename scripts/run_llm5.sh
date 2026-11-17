#!/usr/bin/env bash
# After run_llm4.sh: video LLM + expert-opinion token (vid_exp, seed 1), then the remaining video runs of run_llm3.sh.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -q LLM6_DONE ../outputs/full/logs/run_llm6.log 2>/dev/null; do sleep 120; done   # short adjudicator pilot (run_llm6.sh) goes first
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 20000 ]; do sleep 60; done
ts "start vid_exp seed 1"
LLM_BS=2 nice -n 10 python llm_vid.py --variant vid_exp --seed 1 2>&1 | grep -v "Loading weights"
[ -f ../outputs/full/llm_pilot/vid_exp_seed1.json ] || ts "FAILED vid_exp seed 1"
python llm_analysis.py > /dev/null; ts "LLM5_EXP_DONE"
bash run_llm3.sh
