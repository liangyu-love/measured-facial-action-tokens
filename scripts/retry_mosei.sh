#!/usr/bin/env bash
# Re-run MOSEI seeds that FAILED (OOM while DIFF3 shared the GPU), after DIFF3 ends; refresh the MOSEI summary afterwards.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
export HF_HUB_OFFLINE=1 MSA_DATA=outputs/mosei MSA_DATASET=mosei MSA_RAW=data/mosei/Raw ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-uncased
M=~/work/msa-pilot/outputs/mosei
until grep -q DIFF3_DONE ~/work/msa-pilot/outputs/full/run_diff3.log 2>/dev/null; do sleep 60; done
declare -A SC=([text]="train_bert.py --cond text" [deva_av]="train_bert.py --cond deva_av" [mag]="au_soft.py --variant mag" [soft]="au_soft.py --variant soft" [soft_emo]="au_soft2.py --variant soft_emo" [au_soft_pre]="pre_soft.py --variant au_soft_pre" [diff2_pre]="pre_soft.py --variant diff2_pre")
until grep -q MOSEI_ALL_DONE $M/run_mosei.log 2>/dev/null; do
  grep -h "FAILED" $M/run_mosei.log | awk '{print $4, $5}' | sort -u | while read nm s; do
    [ -n "${SC[$nm]}" ] && python ${SC[$nm]} --seed $s > $M/logs/${nm}_s${s}_retry.log 2>&1 && echo "[$(date '+%F %T')] retried $nm $s"
  done
  sleep 120
done
grep -h "FAILED" $M/run_mosei.log | awk '{print $4, $5}' | sort -u | while read nm s; do
  [ -n "${SC[$nm]}" ] && python ${SC[$nm]} --seed $s > $M/logs/${nm}_s${s}_retry.log 2>&1 && echo "[$(date '+%F %T')] retried $nm $s"
done
python mosi_summary.py > $M/logs/summary.log 2>&1; python pre_analysis.py > $M/logs/pre_analysis.log 2>&1
echo "[$(date '+%F %T')] MOSEI_RETRY_DONE"
