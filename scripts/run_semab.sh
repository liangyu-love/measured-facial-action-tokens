#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
for v in zh_noid zh_code_noid zh_perm_noid; do for s in 1 2 3 4 5; do python xgate_bert.py --variant $v --seed $s > $F/logs/xgate_${v}_s$s.log 2>&1 && echo "$(date +%T) trained $v $s" || echo "$(date +%T) FAILED $v $s"; done; done
python xgate_analysis.py > $F/logs/xgate_analysis.log 2>&1
echo "SEMAB_ALL_DONE $(date +%T)"
