#!/usr/bin/env bash
# Pretrained evidence encoder round: SIMS v2 (in-domain) + MOSI (transfer). MOSEI transfer runs inside run_mosei.sh.
# Starts after the evidence-quality round; writes PRE_DONE (MOSEI GPU stage waits for it). Qwen3-VL stays paused.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -q "EVQ_ALL_DONE" $F/run_evq.log 2>/dev/null; do sleep 30; done
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"
ts "start; encoders: $(ls $F/results_evq/*.pt 2>/dev/null | wc -l)"
run() {  # run <outdir-for-logs> <variant>
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "python pre_soft.py --variant $2 --seed {} > $1/logs/pre_$2_s{}.log 2>&1 || echo FAILED $2 {}"
  ts "$2 done ($MSA_DATASET)"
}
export MSA_DATASET=sims2 MSA_DATA=outputs/full
run $F au_soft_pre; run $F diff2_pre
python pre_analysis.py > $F/logs/pre_analysis.log 2>&1
export MSA_DATASET=mosi MSA_DATA=outputs/mosi ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-uncased
M=~/work/msa-pilot/outputs/mosi
run $M au_soft_pre; run $M diff2_pre
python pre_analysis.py > $M/logs/pre_analysis.log 2>&1
ts "PRE_DONE"
