#!/usr/bin/env bash
# CMU-MOSEI continuation (replaces run_mosei.sh from the AU stage on, after 4/6 AU shards OOMed on high-res frames).
# 3 AU workers over the remaining clips (per-frame fallback on OOM) -> merge -> guard -> emotions -> DEVA text -> training.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
export MSA_DATA=outputs/mosei MSA_RAW=data/mosei/Raw PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AU_MEM_FRAC=0.27
M=~/work/msa-pilot/outputs/mosei
ts() { echo "[$(date '+%F %T')] $*"; }
ts "cont: start AU (3 workers)"
for i in 0 1 2; do python pilot.py au --out $M --clips $M/clips.csv --shard $i/3 > $M/logs/au_c$i.log 2>&1 & done
wait
python pilot.py au --out $M --clips $M/clips.csv > $M/logs/au_merge.log 2>&1 && ts "au merged: $(tail -1 $M/logs/au_merge.log)"
python - <<'PY' || { ts "MOSEI_ABORTED: AU incomplete"; exit 1; }
import os, json, pandas as pd
M = os.path.expanduser("~/work/msa-pilot/outputs/mosei")
c = pd.read_csv(os.path.join(M, "clips.csv"), dtype=str)
framed = {k for k in c.key if os.path.exists(os.path.join(M, "frames", k.replace("/", "__"), "frames.json"))}
a = pd.read_csv(os.path.join(M, "au_frames.csv"), usecols=["key"], dtype=str)
have = set(a.key)
print("clips", len(c), "framed", len(framed), "with AU", len(have & framed), "fallback lines", sum(open(os.path.join(M, "logs", f"au_c{i}.log")).read().count("oom-fallback") for i in range(3)))
assert len(framed - have) == 0, f"{len(framed - have)} framed clips missing AU"
PY
python - <<'PY'
import os, pandas as pd
M = os.path.expanduser("~/work/msa-pilot/outputs/mosei")
a = pd.read_csv(os.path.join(M, "au_frames.csv"))
a[["key", "frame", "face", "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]].to_csv(os.path.join(M, "emo_frames.csv"), index=False)
print("emo rows", len(a))
PY
while pgrep -f "^python prosody_text.py" > /dev/null; do sleep 30; done
DEVA_LANG=en python deva_text.py > $M/logs/deva_text.log 2>&1 && ts "deva_text (en)"
export HF_HUB_OFFLINE=1 MSA_DATASET=mosei ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-uncased
par() { local sc=$1 fl=$2 nm=$3
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "python $sc $fl $nm --seed {} > $M/logs/${nm}_s{}.log 2>&1 && echo \"[\$(date '+%F %T')] trained $nm {}\" || echo \"[\$(date '+%F %T')] FAILED $nm {}\""; }
par train_bert.py --cond text
par train_bert.py --cond deva_av
par au_soft.py --variant mag
par au_soft.py --variant soft
par au_soft2.py --variant soft_emo
par pre_soft.py --variant au_soft_pre
par pre_soft.py --variant diff2_pre
python mosi_summary.py > $M/logs/summary.log 2>&1
python pre_analysis.py > $M/logs/pre_analysis.log 2>&1
ts "MOSEI_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
