#!/usr/bin/env bash
# Frame densification training (after MOSI training + dense AU extraction), 3 seeds in parallel; resumes Qwen3-VL at the end.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -q MOSI_ALL_DONE ~/work/msa-pilot/outputs/mosi/run_train.log 2>/dev/null; do sleep 30; done
while pgrep -f "^python dense_au.py extract" >/dev/null; do sleep 30; done
ts "extraction finished"; python dense_au.py merge > $F/logs/denseau_merge.log 2>&1 && ts "merged: $(tail -1 $F/logs/denseau_merge.log)"
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
for v in crop8 seg8 dense16; do
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "python dense_soft.py --variant $v --seed {} > $F/logs/dense_${v}_s{}.log 2>&1 && echo \"[\$(date '+%F %T')] trained $v {}\" || echo \"[\$(date '+%F %T')] FAILED $v {}\""
done
python dense_analysis.py > $F/logs/dense_analysis.log 2>&1
ts "DENSE_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
