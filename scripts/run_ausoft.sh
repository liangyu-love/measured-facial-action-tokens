#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
until grep -q ROBERTA_ALL_DONE $F/run_roberta.log 2>/dev/null; do sleep 30; done
for v in late soft; do for s in 1 2 3 4 5; do python au_soft.py --variant $v --seed $s > $F/logs/ausoft_${v}_s$s.log 2>&1 && echo "$(date +%T) trained $v $s" || echo "$(date +%T) FAILED $v $s"; done; done
echo "AUSOFT_ALL_DONE $(date +%T)"
