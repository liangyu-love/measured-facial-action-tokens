#!/usr/bin/env bash
# Adjudicator pilot, after run_llm4.sh: the LLM predicts a correction on top of the plain expert average (au_res), plus a 3-way
# closest-expert head (au_adj). Experts use the overall label only (--exp m). Also re-runs LLM + face tokens (seeds 1-5) into llm_val
# so the late-average baseline can pick its weight on the valid split.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2
cd ~/work/msa-pilot/scripts
ts() { echo "[$(date '+%F %T')] $*"; }
gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $1 ]; do sleep 60; done; }
until grep -q LLM4_DONE ../outputs/full/logs/run_llm4.log; do sleep 120; done
dif() { gpu 14000; ts "start $1 --exp m seed $2"; nice -n 10 python llm_diff.py --variant $1 --exp m --seed $2 2>&1 | grep -v "Loading weights"; [ -f ../outputs/full/llm_pilot/$1_m_seed$2.json ] || ts "FAILED $1 seed $2"; }
val() { gpu 14000; ts "start llm_val au seed $1"; LLM_OUT=llm_val nice -n 10 python llm_pilot.py --variant au --seed $1 2>&1 | grep -v "Loading weights"; [ -f ../outputs/full/llm_val/au_seed$1.json ] || ts "FAILED llm_val au seed $1"; }
dif au_adj 1; dif au_res 1; val 1; python llm_adj_analysis.py > /dev/null; ts "LLM6_SEED1_DONE"
for s in 2 3 4 5; do dif au_adj $s; dif au_res $s; val $s; done
python llm_adj_analysis.py
ts "LLM6_DONE"
