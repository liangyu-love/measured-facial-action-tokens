#!/usr/bin/env bash
# Paper-2 pilot C (LLM backbone + measured face tokens). Waits for > 14 GB free GPU memory before every run.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
for s in 1 2 3 4 5; do for v in text au; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 14000 ]; do sleep 60; done
  ts "start $v seed $s"
  nice -n 10 python llm_pilot.py --variant $v --seed $s 2>&1 | grep -v "Loading weights" || ts "FAILED $v seed $s"
  [ $s = 1 ] && [ $v = au ] && ts "LLM_SEED1_DONE"
done; done
python llm_analysis.py
ts "LLM_DONE"
