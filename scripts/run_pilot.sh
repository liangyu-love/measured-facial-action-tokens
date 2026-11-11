#!/usr/bin/env bash
# Go/no-go pilot runner. Usage: nohup bash scripts/run_pilot.sh > outputs/pilot/run.log 2>&1 &
set -euo pipefail
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
P=~/work/msa-pilot/outputs/pilot

# wait for the priority fetch of pilot clips to exit
while pgrep -f "fetch_drive.py fetch --workers 4 --meta outputs/pilot/pilot_clips.csv" >/dev/null; do sleep 20; done
echo "[$(date +%T)] pilot videos: $(python -c "import pandas as pd,os;d=pd.read_csv('$P/pilot_clips.csv',dtype=str);print(sum(os.path.exists(os.path.expanduser('~/work/msa-pilot/data/chsims2/Raw/'+k+'.mp4')) for k in d.key))")/200"

echo "[$(date +%T)] frames";  python pilot.py frames
echo "[$(date +%T)] au";      python pilot.py au
echo "[$(date +%T)] prosody"
python prosody_text.py extract --splits $P/pilot_clips.csv --video-root ~/work/msa-pilot/data/chsims2/Raw --out $P/prosody_feats.csv
python prosody_text.py describe --splits $P/pilot_clips.csv --feats $P/prosody_feats.csv --out $P/prosody_text.jsonl
echo "[$(date +%T)] desc";    HF_HUB_OFFLINE=1 python pilot.py desc
echo "[$(date +%T)] score";   python pilot.py score
echo "[$(date +%T)] done"
