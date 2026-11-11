#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
O=~/work/msa-pilot/outputs/sft3b
until grep -q SCALE_ALL_DONE ../outputs/pilot_scale/run_scale.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) SFT train"; python sft_qwen3b.py > $O/sft_train.log 2>&1 || { echo "SFT FAILED"; exit 1; }
for v in zs sft; do echo "$(date +%T) generate $v"; python gen_full3b.py --variant $v > $O/gen_$v.log 2>&1 || echo "gen $v FAILED"; done
for v in zs sft; do
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_BASE=full SCORE_DESC=$O/$v/desc.jsonl SCORE_KEYS=$O/test_keys.csv SCORE_OUT=$O/$v/score_test python score_v3.py > $O/score_$v.log 2>&1 && echo "$(date +%T) scored $v"
done
for c in vlm3b_zs vlm3b_sft; do for s in 1 2 3 4 5; do
  python train_bert.py --cond $c --seed $s > ../outputs/full/logs/train_${c}_s$s.log 2>&1 && echo "$(date +%T) trained $c $s" || echo "FAILED $c $s"; done; done
python aggregate.py > /dev/null; python sft_eval.py > $O/sft_eval.log 2>&1
echo "SFT_ALL_DONE $(date +%T)"
