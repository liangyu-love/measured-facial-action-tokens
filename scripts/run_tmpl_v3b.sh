#!/usr/bin/env bash
set -uo pipefail
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
while pgrep -f "[d]ense_pose.py --clips" >/dev/null; do sleep 30; done
echo "$(date +%T) dense shards done: $(ls $F/dense/dense_parts | wc -l) clips"
python - <<PY
import os, pandas as pd
p = os.path.expanduser("~/work/msa-pilot/outputs/full/dense/dense_parts")
pd.concat([pd.read_csv(os.path.join(p, f)) for f in sorted(os.listdir(p))]).to_csv(os.path.expanduser("~/work/msa-pilot/outputs/full/dense/dense_pose.csv"), index=False)
PY
python head_actions.py v3b $F/dense/dense_pose.csv $F/dense > $F/dense/head_actions.log 2>&1 && echo "$(date +%T) head actions done"
python measure_text.py --clips $F/clips.csv --au $F/au_frames.csv --out $F/measure_text_v3b.jsonl --head-actions $F/dense/head_actions.jsonl && echo "$(date +%T) template v3b done"
while pgrep -f "[t]rain_bert.py --cond tmpl_vis_shuf" >/dev/null || pgrep -f "[f]or s in 1 2 3 4 5; do python train_bert.py --cond tmpl_vis_shuf" >/dev/null; do sleep 20; done
for s in 1 2 3 4 5; do python train_bert.py --cond tmpl_v3b --seed $s > $F/logs/train_tmpl_v3b_s$s.log 2>&1 && echo "$(date +%T) trained tmpl_v3b $s" || echo "$(date +%T) FAILED tmpl_v3b $s"; done
python aggregate.py > /dev/null
echo "$(date +%T) v3b done"
