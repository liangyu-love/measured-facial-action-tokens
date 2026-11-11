#!/usr/bin/env bash
# CMU-MOSI preprocessing (same pipeline as CH-SIMS v2). Light GPU use (py-feat), can overlap BERT jobs.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
export MSA_DATA=outputs/mosi MSA_RAW=data/mosi/Raw
M=~/work/msa-pilot/outputs/mosi
D=~/work/msa-pilot/data/mosi
mkdir -p $M/logs
ts() { echo "[$(date '+%F %T')] $*"; }
python - <<'PY'
import os, pandas as pd
d = pd.read_csv(os.path.expanduser("~/work/msa-pilot/data/mosi/label.csv"), dtype={"video_id": str, "clip_id": str})
d["key"] = d.video_id + "/" + d.clip_id
d = d.rename(columns={"mode": "split"})
d["category"] = "na"  # MOSI has no unimodal labels -> no Conflict/Align subsets
d["conflict_mods"] = ""
d[["key", "video_id", "clip_id", "split", "text", "label", "label_T", "label_A", "label_V", "category", "conflict_mods"]].to_csv(
    os.path.expanduser("~/work/msa-pilot/outputs/mosi/clips.csv"), index=False)
print(d.split.value_counts().to_dict(), "label range", d.label.min(), d.label.max())
PY
ts "clips.csv"
python pilot.py frames --out $M --clips $M/clips.csv --workers 12 > $M/logs/frames.log 2>&1 && ts "frames"
python pilot.py au --out $M --clips $M/clips.csv > $M/logs/au.log 2>&1 && ts "au"
python emo_frames.py > $M/logs/emo.log 2>&1 && ts "emo"
python prosody_text.py extract --splits $M/clips.csv --video-root $D/Raw --out $M/prosody_feats.csv > $M/logs/prosody.log 2>&1 && ts "prosody"
DEVA_LANG=en python deva_text.py > $M/logs/deva_text.log 2>&1 && ts "deva_text (en)"
[ -s ~/work/msa-pilot/models/bert-base-uncased/model.safetensors ] || \
  HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 \
  hf download google-bert/bert-base-uncased config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors \
  --local-dir ~/work/msa-pilot/models/bert-base-uncased > $M/logs/bert_dl.log 2>&1 && ts "bert-base-uncased"
ts "MOSI_PREP_DONE"
