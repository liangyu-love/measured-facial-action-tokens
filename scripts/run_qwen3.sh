#!/usr/bin/env bash
# Qwen3-VL queue (runs after the gate job). Resumable: every step skips cached work.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs
S=$R/pilot_scale
Q8=$R/full_q3vl8b
SF=$R/sft_q3vl4b
mkdir -p $S $Q8 $SF
ts() { echo "[$(date '+%F %T')] $*"; }
wait_dl() { while pgrep -f "[h]f download Qwen/$1" >/dev/null; do sleep 60; done; }
score_pilot() {  # CPU, background-safe
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_DESC=$S/$1/desc.jsonl SCORE_OUT=$S/$1 python score_v3.py > $S/score_$1.log 2>&1
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_DESC=$S/$1/desc.jsonl SCORE_KEYS=$R/pilot_codex20/selected_clips.csv \
    SCORE_OUT=$S/$1/sub20 python score_v3.py > $S/score_$1_sub20.log 2>&1
}
train() { for c in "$@"; do for s in 1 2 3 4 5; do python train_bert.py --cond $c --seed $s > $R/full/logs/train_${c}_s$s.log 2>&1 \
  && ts "trained $c $s" || ts "FAILED $c $s"; done; done; }

until grep -q GATE_ALL_DONE $R/full/run_gate.log; do sleep 60; done
ts "gate done"
until grep -q ROBERTA_ALL_DONE $R/full/run_roberta.log 2>/dev/null && grep -q AUSOFT_ALL_DONE $R/full/run_ausoft.log 2>/dev/null; do sleep 60; done
ts "roberta done"

# 1) scale check 4B / 8B on the 200 pilot clips (+ re-score existing Qwen2.5-7B on the 20-clip subset)
[ -s $S/7B/sub20/summary_v3b.json ] || score_pilot 7B &
for m in 4B 8B; do
  wait_dl Qwen3-VL-$m-Instruct
  t=q3vl${m,,}
  ts "pilot gen $t"; python scale_desc.py --model Qwen3-VL-$m-Instruct --tag $t > $S/gen_$t.log 2>&1 || ts "FAILED pilot gen $t"
  score_pilot $t &
done

# 2) SFT smoke test (Qwen3-VL-4B, 16 samples) so a broken SFT doesn't block the rest
cp -n $R/sft3b/train.jsonl $R/sft3b/valid.jsonl $R/sft3b/test_keys.csv $SF/ 2>/dev/null
SFT_OK=0
SFT_LIMIT=16 python sft_qwenvl.py > $SF/smoke.log 2>&1 && grep -q "smoke ok" $SF/smoke.log && SFT_OK=1
ts "sft smoke ok=$SFT_OK"

# 3) Qwen3-VL-8B descriptions for all 4403 clips + verified filter (same v2-at-0.5 filter as the 7B vlmv)
ts "full gen 8B"; python gen_full.py --model Qwen3-VL-8B-Instruct --out outputs/full_q3vl8b > $Q8/gen.log 2>&1 || ts "FAILED full gen 8B"
ln -sf $R/full/au_frames.csv $Q8/au_frames.csv
python pilot.py verify --out $Q8 --clips $R/full/clips.csv > $Q8/verify.log 2>&1 && ts "verified 8B"
train vlm8b vlmv8b shuf8b

# 4) SFT ablation on Qwen3-VL-4B
if [ $SFT_OK = 1 ]; then
  ts "SFT train"; python sft_qwenvl.py > $SF/sft_train.log 2>&1 || { ts "SFT FAILED"; SFT_OK=0; }
fi
python gen_full.py --model Qwen3-VL-4B-Instruct --out outputs/sft_q3vl4b/zs > $SF/gen_zs.log 2>&1 && ts "gen 4B zs"
if [ $SFT_OK = 1 ]; then
  python gen_full.py --model Qwen3-VL-4B-Instruct --adapter models/sft_q3vl4b_lora --out outputs/sft_q3vl4b/sft > $SF/gen_sft.log 2>&1 && ts "gen 4B sft"
fi
for v in zs sft; do
  [ -s $SF/$v/desc.jsonl ] || continue
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_BASE=full SCORE_DESC=$SF/$v/desc.jsonl SCORE_KEYS=$SF/test_keys.csv \
    SCORE_OUT=$SF/$v/score_test python score_v3.py > $SF/score_$v.log 2>&1 &
done
train vlm4b_zs; [ $SFT_OK = 1 ] && train vlm4b_sft

# 5) 32B (4-bit NF4) scale check, last because its download is the slowest
wait_dl Qwen3-VL-32B-Instruct
ts "pilot gen q3vl32b"; python scale_desc.py --model Qwen3-VL-32B-Instruct --tag q3vl32b --nf4 > $S/gen_q3vl32b.log 2>&1 || ts "FAILED 32B"
score_pilot q3vl32b
wait
python aggregate.py > /dev/null
SFT_DIR=outputs/sft_q3vl4b python sft_eval.py > $SF/sft_eval.log 2>&1
ts "QWEN3_ALL_DONE"
