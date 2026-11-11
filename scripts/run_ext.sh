#!/usr/bin/env bash
# External experts (no target-dataset unimodal labels): text expert from public corpora + py-feat emotion valence; scalar DIFF
# (diff3_ext) on CH-SIMS v2, MOSI, MOSEI; 5 seeds; audit refresh. VRAM-aware.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs; MD=$HOME/work/msa-pilot/models
ts() { echo "[$(date '+%F %T')] $*"; }
free_ok() { [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 8000 ]; }
ts "start"
for spec in "full sims2 chnsenticorp bert-base-chinese" "mosi mosi sst2 bert-base-uncased" "mosei mosei sst2 bert-base-uncased"; do set -- $spec
  until free_ok; do sleep 30; done
  MSA_DATA=outputs/$1 MSA_DATASET=$2 ENCODER_DIR=$MD/$4 python ext_text.py --corpus $3 > $R/$1/logs/ext_text.log 2>&1 && ts "ext text expert $1: $(grep -E 'corpus valid acc' $R/$1/logs/ext_text.log | tail -1)" || ts "FAILED ext_text $1"
  grep -E "Corr|AUROC" $R/$1/logs/ext_text.log
done
for spec in "full sims2 bert-base-chinese" "mosi mosi bert-base-uncased" "mosei mosei bert-base-uncased"; do set -- $spec
  for s in 1 2 3 4 5; do
    until free_ok; do sleep 30; done
    MSA_DATA=outputs/$1 MSA_DATASET=$2 ENCODER_DIR=$MD/$3 DIFF3_RESULTS_DIR=results_ext python diff3.py --name diff3_ext --precomp ext --seed $s > $R/$1/logs/diff3_ext_s$s.log 2>&1 && ts "diff3_ext $1 $s" || ts "FAILED diff3_ext $1 $s"
  done
done
python audit.py > $R/full/logs/audit_ext.log 2>&1; ts "EXT_DONE"
