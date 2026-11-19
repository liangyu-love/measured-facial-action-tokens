#!/usr/bin/env bash
# Paper-1 addition (2026-11-18): the LLM that sees the 8 frames AND the 8 measured face tokens (llm_vid.py --variant vid_au), seeds 2-5,
# so "frames + tokens" vs "frames only" has 5 seeds. Seed 1 exists from run_llm2.sh. Waits for a free GPU (shared server).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
for s in 2 3 4 5; do
  for b in 2 1; do
    [ -f ../outputs/full/llm_pilot/vid_au_seed$s.json ] && break
    gpu 20000; ts "start vid_au seed $s bs $b"
    LLM_BS=$b nice -n 10 python llm_vid.py --variant vid_au --seed $s 2>&1 | grep -v "Loading weights"
  done
  [ -f ../outputs/full/llm_pilot/vid_au_seed$s.json ] || ts "FAILED vid_au seed $s"
  python llm_analysis.py > /dev/null; ts "LLM8_SEED${s}_DONE"
done
python llm_analysis.py
ts "LLM8_DONE"
