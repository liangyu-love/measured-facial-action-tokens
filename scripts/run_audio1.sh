#!/usr/bin/env bash
# Step 1 of the audio expert: wav extraction (already running) -> HuBERT features -> cross-fitted audio expert; report vs threshold.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATA=outputs/full MSA_DATASET=sims2
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "^python extract_wav.py" > /dev/null; do sleep 30; done
ts "wav: $(tail -1 $F/logs/extract_wav.log)"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 4000 ]; do sleep 30; done
python hubert_feats.py > $F/logs/hubert_feats.log 2>&1 && ts "hubert: $(tail -1 $F/logs/hubert_feats.log)" || { ts "FAILED hubert"; exit 1; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 3000 ]; do sleep 30; done
python audio_expert_hubert.py > $F/logs/audio_expert_hubert.log 2>&1; ts "$(grep MEAN $F/logs/audio_expert_hubert.log)"
ts "AUDIO1_DONE"
