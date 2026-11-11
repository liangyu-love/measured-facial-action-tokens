#!/usr/bin/env bash
# Uniform tuning round. A) after MOSEI first round: SIMS v2 + MOSI (stage 1 grid on valid, stage 2 five seeds)
# B) Qwen3-VL queue overnight  C) MOSEI reduced-grid tuning after Qwen3-VL finishes.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs
ts() { echo "[$(date '+%F %T')] $*"; }
pause_q() { pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"; }
until grep -q MOSEI_ALL_DONE $R/mosei/run_mosei.log 2>/dev/null; do sleep 60; done
until grep -q DIFF3_DONE $R/full/run_diff3.log 2>/dev/null; do sleep 60; done
sleep 15; pause_q; ts "A: qwen3 paused"
# smoke: one valid-only epoch into a throwaway dir
HF_HUB_OFFLINE=1 RESULTS_DIR=tune_smoke python train_bert.py --cond text --seed 9 --epochs 1 --lr 2e-5 --lr_new 1e-3 --loss l1 --patience 3 --valid_only \
  > $R/full/logs/tune_smoke.log 2>&1 && ts "smoke ok: $(tail -1 $R/full/logs/tune_smoke.log | cut -c1-80)" || { ts "SMOKE FAILED"; exit 1; }
rm -rf $R/full/tune_smoke
# expert tuning on valid (label_V / label_T), then tuned experts for seeds 1-10 (used by all expert-based methods)
python expert_tune.py > $R/full/logs/expert_tune.log 2>&1; ts "expert tuning: $(tail -2 $R/full/logs/expert_tune.log | tr '
bash ~/work/msa-pilot/scripts/run_pretune.sh > $R/full/run_pretune.log 2>&1; ts "pretune: $(grep -E 'backbone|PRETUNE' $R/full/run_pretune.log | tail -2 | tr -d '\n')"
' ' ')"
# best densified variant chosen on VALID (mean valid MAE over its 5 seeds)
DV=$(python - <<'PY'
import glob, json, os, numpy as np
F = os.path.expanduser("~/work/msa-pilot/outputs/full/results_dense")
best = min(("crop8", "seg8", "dense16"), key=lambda v: np.mean([json.load(open(p))["valid"]["mae"] for p in glob.glob(f"{F}/{v}_seed*.json")] or [9]))
print(best)
PY
)
ts "densified variant (by valid MAE): $DV"
python tune.py stage1 --dataset sims2 --workers 3 --dense_variant $DV > $R/full/logs/tune_stage1.log 2>&1 && ts "sims2 stage1"
python tune.py stage1 --dataset mosi --workers 3 > $R/mosi/logs/tune_stage1.log 2>&1 && ts "mosi stage1"
python tune.py stage2 --dataset sims2 --workers 3 --dense_variant $DV > $R/full/logs/tune_stage2.log 2>&1 && ts "sims2 stage2"
python tune.py stage2 --dataset mosi --workers 3 > $R/mosi/logs/tune_stage2.log 2>&1 && ts "mosi stage2"
# 10-seed check: seeds 6-10 for ALL SIMS methods under their tuned configs (no selective seed adding)
python tune.py stage2 --dataset sims2 --workers 3 --dense_variant $DV --seed_from 6 --seed_to 10 > $R/full/logs/tune_stage2_s6_10.log 2>&1 && ts "sims2 stage2 seeds 6-10"
python tune.py summary --dataset sims2 --dense_variant $DV > /dev/null 2>&1; python tune.py summary --dataset mosi > /dev/null 2>&1
MSA_DATASET=sims2 MSA_DATA=outputs/full python tune_sig.py > $R/full/logs/tune_sig.log 2>&1; MSA_DATASET=mosi MSA_DATA=outputs/mosi python tune_sig.py > $R/mosi/logs/tune_sig.log 2>&1
ts "TUNE_A_DONE"
# A2) stronger text encoder on MOSI (all methods) before Qwen3-VL
bash ~/work/msa-pilot/scripts/run_deberta.sh >> $R/mosi/run_deberta.log 2>&1; ts "A2: deberta MOSI: $(tail -1 $R/mosi/run_deberta.log)"
# B) Qwen3-VL overnight
cd ~/work/msa-pilot && setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &
cd ~/work/msa-pilot/scripts; ts "B: qwen3 resumed"
until grep -q QWEN3_ALL_DONE $R/run_qwen3.log 2>/dev/null; do sleep 120; done
ts "C: qwen3 finished"
python tune.py stage1 --dataset mosei --workers 3 > $R/mosei/logs/tune_stage1.log 2>&1 && ts "mosei stage1"
python tune.py stage2 --dataset mosei --workers 3 > $R/mosei/logs/tune_stage2.log 2>&1 && ts "mosei stage2"
python tune.py summary --dataset mosei > /dev/null 2>&1
ts "TUNE_ALL_DONE"
