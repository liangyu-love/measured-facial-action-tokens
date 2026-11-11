#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
for c in deva_v deva_av; do for s in 1 2 3 4 5; do
  python train_bert.py --cond $c --seed $s > $F/logs/train_${c}_s$s.log 2>&1 && echo "$(date +%T) trained $c $s" || echo "$(date +%T) FAILED $c $s"; done; done
for v in nogate gate gate_aux; do for s in 1 2 3 4 5; do
  python gate_bert.py --variant $v --seed $s > $F/logs/gate_${v}_s$s.log 2>&1 && echo "$(date +%T) trained gate $v $s" || echo "$(date +%T) FAILED gate $v $s"; done; done
python gate_analysis.py > $F/logs/gate_analysis.log 2>&1
echo "GATE_ALL_DONE $(date +%T)"
