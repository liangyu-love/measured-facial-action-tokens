#!/usr/bin/env bash
# Re-run of the densification training after the shard-0 OOM fix (all 4403 clips). Does NOT resume Qwen3-VL
# (the MOSEI queue does that at its end).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "^python dense_au.py extract" >/dev/null; do sleep 30; done
python dense_au.py merge > $F/logs/denseau_merge2.log 2>&1; ts "merged: $(tail -1 $F/logs/denseau_merge2.log)"
n=$(python -c "import pandas as pd;print(pd.read_csv('$F/dense_au.csv',usecols=['key']).key.nunique())")
if [ "$n" != "4403" ]; then ts "DENSE2_ABORTED: only $n clips"; exit 1; fi
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 paused"
for v in crop8 seg8 dense16; do
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "python dense_soft.py --variant $v --seed {} > $F/logs/dense_${v}_s{}.log 2>&1 && echo \"[\$(date '+%F %T')] trained $v {}\" || echo \"[\$(date '+%F %T')] FAILED $v {}\""
done
python dense_analysis.py > $F/logs/dense_analysis.log 2>&1
ts "DENSE2_ALL_DONE"
