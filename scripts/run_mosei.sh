#!/usr/bin/env bash
# CMU-MOSEI: same pipeline as MOSI (8 frames, py-feat AU + emotions in ONE pass, 6 shards), then training
# (text / DEVA-style / MAG-style / au_soft / au_soft+emo, bert-base-uncased, 5 seeds, 3 in parallel). Resumes Qwen3-VL at the end.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
export MSA_DATA=outputs/mosei MSA_RAW=data/mosei/Raw
M=~/work/msa-pilot/outputs/mosei
D=~/work/msa-pilot/data/mosei
mkdir -p $M/logs
ts() { echo "[$(date '+%F %T')] $*"; }
until [ -f $D/UPLOAD_OK ]; do sleep 30; done
if [ ! -d $D/Raw ]; then unzip -q -o $D/Raw.zip -d $D && ts "unzipped"; fi
python - <<'PY'
import os, pandas as pd
d = pd.read_csv(os.path.expanduser("~/work/msa-pilot/data/mosei/label.csv"), dtype={"video_id": str, "clip_id": str})
d["key"] = d.video_id + "/" + d.clip_id
d = d.rename(columns={"mode": "split"})
raw = os.path.expanduser("~/work/msa-pilot/data/mosei/Raw")
ok = [os.path.exists(os.path.join(raw, f"{k}.mp4")) for k in d.key]
print("labeled", len(d), "with video", sum(ok))
d = d[ok]
d["category"] = "na"; d["conflict_mods"] = ""
d[["key", "video_id", "clip_id", "split", "text", "label", "label_T", "label_A", "label_V", "category", "conflict_mods"]].to_csv(
    os.path.expanduser("~/work/msa-pilot/outputs/mosei/clips.csv"), index=False)
print(d.split.value_counts().to_dict())
PY
ts "clips.csv"
python pilot.py frames --out $M --clips $M/clips.csv --workers 14 > $M/logs/frames.log 2>&1 && ts "frames"
( python prosody_text.py extract --splits $M/clips.csv --video-root $D/Raw --out $M/prosody_feats.csv > $M/logs/prosody.log 2>&1 && ts "prosody" ) &
PROS=$!
# GPU stage only after the SIMS/MOSI pretrained-evidence round (run_pre: after evq, after densification)
until grep -q "PRE_DONE" ~/work/msa-pilot/outputs/full/run_pre.log 2>/dev/null; do sleep 30; done
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 paused"
for i in 0 1 2 3 4 5; do python pilot.py au --out $M --clips $M/clips.csv --shard $i/6 > $M/logs/au_s$i.log 2>&1 & done
sleep 30
while pgrep -f "^python pilot.py au" >/dev/null; do sleep 20; done
python pilot.py au --out $M --clips $M/clips.csv > $M/logs/au_merge.log 2>&1 && ts "au merged: $(tail -1 $M/logs/au_merge.log)"
python - <<'PY'
import os, pandas as pd
M = os.path.expanduser("~/work/msa-pilot/outputs/mosei")
a = pd.read_csv(os.path.join(M, "au_frames.csv"))
a[["key", "frame", "face", "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]].to_csv(os.path.join(M, "emo_frames.csv"), index=False)
print("emo rows", len(a))
PY
wait $PROS
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
ts "MOSEI_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
