#!/usr/bin/env bash
# Paper-2 pilot C part 4: decision controls. Waits for free GPU memory before every run.
#  (1) au_spk: face tokens + one speech token, no experts            (is the gain just "speech was added"?)
#  (2) experts trained on the overall label only (oof_experts_m.py)  (does the gain need unimodal labels?)
#  (3) seeds 6-10 of au / au_tri_trust / its capacity control        (firm up the comparison with the control)
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
dif() { gpu 14000; ts "start $*"; nice -n 10 python llm_diff.py --variant "$@" 2>&1 | grep -v "Loading weights" || ts "FAILED $*"; }
for s in 1 2 3 4 5; do dif au_spk --seed $s; done; python llm_analysis.py > /dev/null; ts "LLM4_SPK_DONE"
for s in 1 2 3 4 5; do gpu 8000; ts "start experts_m seed $s"; nice -n 10 python oof_experts_m.py --seed $s 2>&1 | grep -E "Corr|Error|error|Traceback" || ts "FAILED experts_m $s"; done
for s in 1 2 3 4 5; do dif au_tri_trust --exp m --seed $s; dif au_tri_trust --exp m --vctl zero --seed $s; done; python llm_analysis.py > /dev/null; ts "LLM4_NOLAB_DONE"
for s in 6 7 8 9 10; do
  gpu 14000; ts "start au seed $s"; nice -n 10 python llm_pilot.py --variant au --seed $s 2>&1 | grep -v "Loading weights" || ts "FAILED au $s"
  dif au_tri_trust --seed $s; dif au_tri_trust --vctl zero --seed $s
done
python llm_analysis.py
ts "LLM4_DONE"
