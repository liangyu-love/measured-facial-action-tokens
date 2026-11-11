#!/usr/bin/env bash
# DIFF v2 (after the accuracy round, before Qwen3-VL). Pauses & resumes the Qwen3-VL queue.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -q ACC_ALL_DONE $F/run_acc.log 2>/dev/null; do sleep 20; done
sleep 10; pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 paused"
STAT=""
python acc_analysis.py --combo | grep -qw stat && STAT="--stat"
ts "use stat token: [${STAT:-no}]"
python diff2.py --variant diff2_full --seed 0 --epochs 1 $STAT > $F/logs/diff2_smoke.log 2>&1 && ts "smoke ok" || { ts "SMOKE FAILED"; }
rm -f $F/results_diff2/*_seed0*
for v in diff2_nosup diff2_uni diff2_full; do for s in 1 2 3 4 5; do
  python diff2.py --variant $v --seed $s $STAT > $F/logs/diff2_${v}_s$s.log 2>&1 && ts "trained $v $s" || ts "FAILED $v $s"; done; done
python diff2_analysis.py > $F/logs/diff2_analysis.log 2>&1
ts "DIFF2_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
