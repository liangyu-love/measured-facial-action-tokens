#!/usr/bin/env bash
# Paper-2 pilot C parts 2 + 3, one queue. Every run waits for enough free GPU memory, so it never competes with other users.
#  (c) paper-1 analysis modules on the LLM: discrepancy token, speech expert, whom-to-trust head, capacity control   [llm_diff.py]
#  (a) same LLM seeing the real 8 frames, with / without measured face tokens                                        [llm_vid.py]
#  (b) 10-epoch version of text / au                                                                                [llm_pilot.py]
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
dif() { gpu 14000; ts "start $1 $3 seed $2"; nice -n 10 python llm_diff.py --variant $1 --seed $2 $3 2>&1 | grep -v "Loading weights" || ts "FAILED $1 $3 seed $2"; }
vid() { gpu 18000; ts "start $1 seed $2"; nice -n 10 python llm_vid.py --variant $1 --seed $2 2>&1 | grep -v "Loading weights" || ts "FAILED $1 seed $2"; }
txt() { gpu 14000; ts "start ep10 $1 seed $2"; LLM_OUT=llm_ep10 nice -n 10 python llm_pilot.py --variant $1 --seed $2 --epochs 10 2>&1 | grep -v "Loading weights" || ts "FAILED ep10 $1 seed $2"; }
difseed() { dif au_tri_trust $1; dif au_tri_trust $1 "--vctl zero"; dif au_diff $1; dif au_tri $1; }
difseed 1; python llm_analysis.py > /dev/null; ts "LLM2_DIFF_SEED1_DONE"
vid vid 1; vid vid_au 1; python llm_analysis.py > /dev/null; ts "LLM2_VID_SEED1_DONE"
for s in 2 3 4 5; do difseed $s; done; python llm_analysis.py > /dev/null; ts "LLM2_DIFF_DONE"
for s in 1 2 3 4 5; do txt au $s; txt text $s; done; python llm_analysis.py > /dev/null; ts "LLM2_EP10_DONE"
for s in 2 3; do vid vid $s; vid vid_au $s; done
python llm_analysis.py
ts "LLM2_DONE"
