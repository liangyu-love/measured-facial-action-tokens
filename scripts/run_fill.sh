#!/usr/bin/env bash
# Planned fill-ins (11-09): 32B scale check with the GPU to itself; then missing seeds dreg s3 / trust01 s1; then MOSI MAG-style+DeBERTa diagnosis.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs; S=$R/pilot_scale; F=$R/full; M=$R/mosi
ts() { echo "[$(date "+%F %T")] $*"; }
ts "start; GPU free MB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
python scale_desc.py --model Qwen3-VL-32B-Instruct --tag q3vl32b --nf4 > $S/gen_q3vl32b.log 2>&1 && ts "32B gen ok" || ts "FAILED 32B gen"
( CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_DESC=$S/q3vl32b/desc.jsonl SCORE_OUT=$S/q3vl32b python score_v3.py > $S/score_q3vl32b.log 2>&1
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_DESC=$S/q3vl32b/desc.jsonl SCORE_KEYS=$R/pilot_codex20/selected_clips.csv SCORE_OUT=$S/q3vl32b/sub20 python score_v3.py > $S/score_q3vl32b_sub20.log 2>&1
  python aggregate.py > /dev/null 2>&1; ts "32B scored" ) &
export HF_HUB_OFFLINE=1
( MSA_DATASET=sims2 MSA_DATA=outputs/full EXPERT_DIR_V=results_evq_tuned DIFF3_RESULTS_DIR=results_oof python diff3.py --name diff3_reg --precomp reg --seed 3 > $F/logs/fill_dreg_s3.log 2>&1 && ts "dreg s3 ok" || ts "FAILED dreg s3" ) &
( MSA_DATASET=sims2 MSA_DATA=outputs/full EXPERT_DIR_V=results_evq_tuned ENCODER_DIR=$HOME/work/msa-pilot/models/bert-base-chinese EXPERT_DIR_T=results_texp_tuned DIFF3_RESULTS_DIR=results_bert2 python diff3.py --name diff3_trust01 --texp --unc --trust 0.1 --seed 1 > $F/logs/fill_trust01_s1.log 2>&1 && ts "trust01 s1 ok" || ts "FAILED trust01 s1" ) &
wait
export MSA_DATA=outputs/mosi MSA_DATASET=mosi ENCODER_DIR=$HOME/work/msa-pilot/models/deberta-v3-base-fp32
# (a) same recipe, fresh LayerNorm skipped; (b) original MAG, 12 epochs (is it just slow?). 5 seeds each, separate dirs.
printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "MAG_NOLN=1 AUSOFT_RESULTS_DIR=results_deberta_mag_noln python au_soft.py --variant mag --seed {} > $M/logs/deberta_mag_noln_s{}.log 2>&1 || echo FAILED noln {}"
ts "mag noln done"
printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "AUSOFT_RESULTS_DIR=results_deberta_mag_e12 python au_soft.py --variant mag --epochs 12 --seed {} > $M/logs/deberta_mag_e12_s{}.log 2>&1 || echo FAILED e12 {}"
ts "mag e12 done"
python audit.py > $F/logs/audit_fill.log 2>&1
ts "FILL_DONE"
