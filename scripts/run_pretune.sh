#!/usr/bin/env bash
# Before uniform tuning (called by run_tune.sh after expert_tune.py): (1) RoBERTa-wwm-ext backbone swap, (2) audio expert diff3_tri.
# 5 seeds each; experts = tuned visual expert; text expert = tuned config (BERT or RoBERTa to match the backbone).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full EXPERT_DIR_V=results_evq_tuned
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full; MD=$HOME/work/msa-pilot/models
ROB=$MD/chinese-roberta-wwm-ext; BRT=$MD/bert-base-chinese
ts() { echo "[$(date '+%F %T')] $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
sched() { local pids=()
  while IFS= read -r line; do local lg=${line%%|*} cmd=${line#*|}
    while :; do pids=($(for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && echo $p; done)); [ ${#pids[@]} -lt 3 ] && [ "$(free_mb)" -gt 7000 ] && break; sleep 15; done
    ( eval "$cmd" > $F/logs/$lg.log 2>&1 || echo "[$(date '+%F %T')] FAILED $lg" ) & pids+=($!); sleep 30
  done < "$1"; wait; }
TL=$(python pretune_cfg.py lr); TE=$(python pretune_cfg.py epochs)
ts "start; tuned text expert cfg lr=$TL epochs=$TE"
J=$(mktemp)
: > $J
for s in 1 2 3 4 5; do echo "texp_rob_s$s|ENCODER_DIR=$ROB EXPERT_DIR_T=results_texp_roberta python texp.py --seed $s --lr $TL --epochs $TE" >> $J; done
for s in $(seq 1 10); do echo "aexp_s$s|python aexp.py --seed $s" >> $J; done
sched $J; ts "experts done: rob $(ls $F/results_texp_roberta/*.pt | wc -l), audio $(ls $F/results_aexp/*.pt | wc -l)"
: > $J
for s in 1 2 3 4 5; do
  echo "rob_text_s$s|ENCODER_DIR=$ROB RESULTS_DIR=results_rob2 python train_bert.py --cond text --seed $s" >> $J
  echo "rob_emo_s$s|ENCODER_DIR=$ROB AUSOFT2_RESULTS_DIR=results_rob2 python au_soft2.py --variant soft_emo --seed $s" >> $J
  echo "rob_d2_s$s|ENCODER_DIR=$ROB DIFF3_RESULTS_DIR=results_rob2 python diff3.py --name diff2_pre --seed $s" >> $J
  echo "rob_unc_s$s|ENCODER_DIR=$ROB EXPERT_DIR_T=results_texp_roberta DIFF3_RESULTS_DIR=results_rob2 python diff3.py --name diff3_unc --texp --unc --seed $s" >> $J
  echo "brt_d2_s$s|ENCODER_DIR=$BRT DIFF3_RESULTS_DIR=results_bert2 python diff3.py --name diff2_pre --seed $s" >> $J
  echo "brt_unc_s$s|ENCODER_DIR=$BRT EXPERT_DIR_T=results_texp_tuned DIFF3_RESULTS_DIR=results_bert2 python diff3.py --name diff3_unc --texp --unc --seed $s" >> $J
  echo "tri_s$s|ENCODER_DIR=$BRT EXPERT_DIR_T=results_texp_tuned DIFF3_RESULTS_DIR=results_bert2 python diff3.py --name diff3_tri --texp --unc --aexp --seed $s" >> $J
  # (3) whom-to-trust head (target |y-y_T|-|y-y_V|, train only), lambda 0.1 / 0.3
  echo "trust01_s$s|ENCODER_DIR=$BRT EXPERT_DIR_T=results_texp_tuned DIFF3_RESULTS_DIR=results_bert2 python diff3.py --name diff3_trust01 --texp --unc --trust 0.1 --seed $s" >> $J
  echo "trust03_s$s|ENCODER_DIR=$BRT EXPERT_DIR_T=results_texp_tuned DIFF3_RESULTS_DIR=results_bert2 python diff3.py --name diff3_trust03 --texp --unc --trust 0.3 --seed $s" >> $J
done
sched $J; ts "runs done"
python pretune_analysis.py > $F/logs/pretune_analysis.log 2>&1; ts "analysis: $(grep -E '^BACKBONE' $F/logs/pretune_analysis.log)"
python trust_decide.py > $F/logs/trust_decide.log 2>&1; ts "trust: $(tail -1 $F/logs/trust_decide.log)"
python audit.py > $F/logs/audit_pretune.log 2>&1; ts "audit refreshed"
if grep -q "^BACKBONE roberta" $F/logs/pretune_analysis.log; then
  : > $J; for s in 6 7 8 9 10; do echo "texp_rob_s$s|ENCODER_DIR=$ROB EXPERT_DIR_T=results_texp_roberta python texp.py --seed $s --lr $TL --epochs $TE" >> $J; done
  sched $J; echo roberta > $F/sims_backbone.txt; ts "SIMS backbone -> RoBERTa (text experts seeds 6-10 trained)"
else echo bert > $F/sims_backbone.txt; ts "SIMS backbone stays bert-base-chinese"; fi
ts "PRETUNE_DONE"
