#!/usr/bin/env bash
# Re-run MOSEI DEVA-style (deva_av) after the tokenizer fix; one seed at a time, only when >7 GB VRAM is free.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
export HF_HUB_OFFLINE=1 MSA_DATA=outputs/mosei MSA_DATASET=mosei ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-uncased
M=~/work/msa-pilot/outputs/mosei
for s in 1 2 3 4 5; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 7000 ]; do sleep 30; done
  python train_bert.py --cond deva_av --seed $s > $M/logs/deva_av_s${s}_fix.log 2>&1 && echo "[$(date '+%F %T')] trained deva_av $s" || echo "[$(date '+%F %T')] FAILED deva_av $s"
done
python mosi_summary.py > $M/logs/summary.log 2>&1; python audit.py > ~/work/msa-pilot/outputs/full/logs/audit_deva.log 2>&1
echo "[$(date '+%F %T')] MOSEI_DEVA_DONE"
