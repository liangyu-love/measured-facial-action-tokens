#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
python clip_feats.py > $F/logs/clip_feats.log 2>&1 || { echo CLIP_FAILED; exit 1; }
for s in 1 2 3 4 5; do AUSOFT_RESULTS_DIR=results_ausoft python au_soft.py --variant soft_clip --seed $s > $F/logs/core_soft_clip_s$s.log 2>&1 && echo "$(date +%T) trained soft_clip $s" || echo "FAILED soft_clip $s"; done
python sig_tests.py > $F/logs/sig_tests.log 2>&1
echo CLIP_ALL_DONE
