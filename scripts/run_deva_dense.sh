#!/usr/bin/env bash
# Faithful DEVA-style baseline: build 10 fps descriptions (k in 3/5/8) -> pick k on VALID (default recipe, seed 1) -> uniform tuning
# (same grid) -> 10 test seeds -> audit. Runs alongside the Qwen3 queue (one job at a time when >6 GB VRAM free).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
free_ok() { [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 6000 ]; }
python deva_dense.py > $F/logs/deva_dense.log 2>&1 && ts "descriptions: $(head -1 $F/logs/deva_dense.log)" || { ts "FAILED deva_dense"; exit 1; }
for k in 3 5 8; do
  until free_ok; do sleep 30; done
  DEVA_FILE=deva_text_dense_k$k.jsonl RESULTS_DIR=deva_k_select/k$k python train_bert.py --cond deva_av --seed 1 --valid_only --patience 0 > $F/logs/deva_k$k.log 2>&1 || ts "FAILED k$k"
done
K=$(python -c "
import json, glob
r = {k: json.load(open(glob.glob('$F/deva_k_select/k%d/*_seed1.json' % k)[0]))['valid']['mae'] for k in (3, 5, 8)}
print(min(r, key=r.get))")
echo $K > $F/deva_dense_k.txt; ts "k selected on valid: $K"
until free_ok; do sleep 30; done
python tune.py stage1 --dataset sims2 --workers 2 --dense_variant crop8 > $F/logs/tune_stage1_deva_dense.log 2>&1 && ts "stage1"
python tune.py stage2 --dataset sims2 --workers 2 --dense_variant crop8 > $F/logs/tune_stage2_deva_dense.log 2>&1 && ts "stage2 1-5"
python tune.py stage2 --dataset sims2 --workers 2 --dense_variant crop8 --seed_from 6 --seed_to 10 > $F/logs/tune_stage2_deva_dense_6_10.log 2>&1 && ts "stage2 6-10"
python peek_final.py > $F/logs/peek_final_deva.log 2>&1; ts "DEVA_DENSE_DONE"
