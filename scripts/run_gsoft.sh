#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
until grep -q EVI_ALL_DONE $F/run_evi.log 2>/dev/null; do sleep 20; done
sleep 5; pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"; echo "$(date +%T) qwen3 paused"
for v in soft_gate soft_gate_aux soft_tokgate; do for s in 1 2 3 4 5; do python gate_soft.py --variant $v --seed $s > $F/logs/gsoft_${v}_s$s.log 2>&1 && echo "$(date +%T) trained $v $s" || echo "$(date +%T) FAILED $v $s"; done; done
python gate_soft_analysis.py > $F/logs/gsoft_analysis.log 2>&1
echo "GSOFT_ALL_DONE $(date +%T)"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
echo "qwen3 resumed"
