#!/usr/bin/env bash
# Replaces run_llm8.sh (2026-11-19): LLM sees 8 frames + 8 measured face tokens (llm_vid.py --variant vid_au), seeds 2-5.
# llm_vid.py now checkpoints every epoch; on a CUDA OOM (another user's job took memory) this waits for free memory and resumes,
# first at bs 2 then at bs 1 (effective batch 16 either way). Any other error stops that seed (FAILED) instead of looping.
# Lines are flushed as they come (PYTHONUNBUFFERED + grep --line-buffered), so progress is visible in the log.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
P=../outputs/full/llm_pilot
# let the seed-2 run started by run_llm8.sh finish (old code, no checkpoint); if it dies, the loop below redoes seed 2
while pgrep -f "llm_vid.py --variant vid_au" > /dev/null; do sleep 60; done
for s in 2 3 4 5; do
  b=2
  for try in $(seq 1 30); do
    [ -f $P/vid_au_seed$s.json ] && break
    gpu 20000; ts "start vid_au seed $s bs $b (try $try)"
    tmp=$(mktemp)
    LLM_BS=$b nice -n 10 python llm_vid.py --variant vid_au --seed $s 2>&1 | grep --line-buffered -v "Loading weights" | tee $tmp
    if [ ! -f $P/vid_au_seed$s.json ]; then
      if grep -q "OutOfMemoryError\|CUDA out of memory" $tmp; then ts "OOM on seed $s, will resume"; b=1; sleep 120
      else rm -f $tmp; break; fi
    fi
    rm -f $tmp
  done
  [ -f $P/vid_au_seed$s.json ] || ts "FAILED vid_au seed $s"
  python llm_analysis.py > /dev/null; ts "LLM9_SEED${s}_DONE"
done
python llm_analysis.py
ts "LLM9_DONE"
