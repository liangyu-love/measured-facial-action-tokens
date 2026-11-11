#!/usr/bin/env bash
# MOSI capacity controls for diff3_ext: visual reading zeroed / shuffled; text-only + SST-2 text expert (no AU tokens). 5 seeds.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATA=outputs/mosi MSA_DATASET=mosi ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-uncased DIFF3_RESULTS_DIR=results_ext
cd ~/work/msa-pilot/scripts
M=~/work/msa-pilot/outputs/mosi
ts() { echo "[$(date '+%F %T')] $*"; }
free_ok() { [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 8000 ]; }
for s in 1 2 3 4 5; do
  for v in "ext_vzero --vctl zero" "ext_vshuf --vctl shuf" "ext_textonly --vctl zero --noau"; do set -- $v; nm=$1; shift
    until free_ok; do sleep 30; done
    python diff3.py --name $nm --precomp ext --seed $s "$@" > $M/logs/${nm}_s$s.log 2>&1 && ts "$nm $s" || ts "FAILED $nm $s"
  done
done
python audit.py > ~/work/msa-pilot/outputs/full/logs/audit_extcap.log 2>&1; ts "EXT_CAPCTL_DONE"
