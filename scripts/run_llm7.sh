#!/usr/bin/env bash
# Paper-1 addition: the LLM that sees the real 8 frames (llm_vid.py --variant vid), seeds 3-5, so the comparison with measured face
# tokens has 5 seeds. Replaces run_llm5.sh (vid_exp and extra vid_au seeds were cancelled, user decision 2026-11-16).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
for s in 3 4 5; do
  for b in 2 1; do
    [ -f ../outputs/full/llm_pilot/vid_seed$s.json ] && break
    gpu 20000; ts "start vid seed $s bs $b"
    LLM_BS=$b nice -n 10 python llm_vid.py --variant vid --seed $s 2>&1 | grep -v "Loading weights"
  done
  [ -f ../outputs/full/llm_pilot/vid_seed$s.json ] || ts "FAILED vid seed $s"
  python llm_analysis.py > /dev/null; ts "LLM7_SEED${s}_DONE"
done
python llm_analysis.py
ts "LLM7_DONE"
