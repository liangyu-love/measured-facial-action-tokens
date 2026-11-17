#!/usr/bin/env bash
# Re-run of the video comparison (llm_vid.py), which ran out of memory at batch 4 in run_llm2.sh. Tries batch 2, falls back to batch 1.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
vid() {
  for b in 2 1; do
    [ -f ../outputs/full/llm_pilot/$1_seed$2.json ] && return
    gpu 20000; ts "start $1 seed $2 bs $b"
    LLM_BS=$b nice -n 10 python llm_vid.py --variant $1 --seed $2 2>&1 | grep -v "Loading weights"
  done
  [ -f ../outputs/full/llm_pilot/$1_seed$2.json ] || ts "FAILED $1 seed $2"
}
for s in 1 2 3; do vid vid $s; vid vid_au $s; python llm_analysis.py > /dev/null; ts "LLM3_SEED${s}_DONE"; done
python llm_analysis.py
ts "LLM3_DONE"
