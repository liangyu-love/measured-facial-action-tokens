#!/usr/bin/env bash
# Full downstream test on all 4403 labeled clips. Resumable: every stage skips cached work.
# Usage: setsid nohup bash scripts/run_full.sh > outputs/full/run.log 2>&1 < /dev/null &
set -uo pipefail
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
C=$F/clips.csv
RAW=~/work/msa-pilot/data/chsims2/Raw
SEEDS="1 2 3 4 5"
ts() { echo "[$(date +%F' '%T)] $*"; }
train() { for c in "$@"; do for s in $SEEDS; do python train_bert.py --cond $c --seed $s > $F/logs/train_${c}_s${s}.log 2>&1 \
          && ts "trained $c seed $s" || ts "FAILED $c seed $s (see logs/train_${c}_s${s}.log)"; done; done; }
mkdir -p $F/logs

[ -f $C ] || python make_clips.py
ts "text-only training (background, overlaps frames/AU)"
( train text ) &
T1=$!

ts "frames";  python pilot.py frames --out $F --clips $C --workers 12 || ts "frames had errors"
ts "prosody extract (background, CPU)"
( python prosody_text.py extract --splits $C --video-root $RAW --out $F/prosody_feats.csv > $F/logs/prosody.log 2>&1 \
  && python prosody_text.py describe --splits $C --feats $F/prosody_feats.csv --out $F/prosody_text.jsonl >> $F/logs/prosody.log 2>&1 \
  && ts "prosody done" || ts "prosody FAILED" ) &
P1=$!
ts "au";      python pilot.py au --out $F --clips $C || ts "au had errors"
wait $P1
ts "measure template"; python measure_text.py --clips $C --au $F/au_frames.csv --out $F/measure_text.jsonl

wait $T1
ts "template training (background, overlaps Qwen)"
( train tmpl ) &
T2=$!
ts "qwen desc"; python pilot.py desc --out $F --clips $C || ts "desc had errors"
ts "verify";    python pilot.py verify --out $F --clips $C
wait $T2
ts "VLM-condition training"
train vlm vlmv tmpl_vlmv shuf
ts "aggregate"; python aggregate.py
ts "done"
