#!/usr/bin/env bash
# Accuracy round (after EVI round 3, before Qwen3-VL): A, B, C, E(combo), F, then D/D0. Pauses & resumes Qwen3-VL.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
run() { local v=$1; for s in 1 2 3 4 5; do python au_soft2.py --variant $v --seed $s > $F/logs/acc_${v}_s$s.log 2>&1 \
  && ts "trained $v $s" || ts "FAILED $v $s"; done; }
until grep -qE "EVI3_ALL_DONE|EVI3_ABORTED" $F/run_evi3.log 2>/dev/null; do sleep 30; done
sleep 10; pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 paused"
# smoke (seed 0, 1 epoch) for the variants that don't need emotion data
for v in soft_stat soft_vtaux03 soft_diff xattn_pre selfattn_pre; do
  python au_soft2.py --variant $v --seed 0 --epochs 1 > $F/logs/acc_smoke_$v.log 2>&1 && ts "smoke ok $v" || ts "SMOKE FAILED $v"; done
rm -f $F/results_acc/*_seed0*; rm -rf $F/xattn_maps
run soft_stat                                     # A
for v in soft_vaux01 soft_vaux03 soft_vtaux03; do run $v; done   # B
until [ -s $F/emo_frames.csv ] && ! pgrep -f "^python emo_frames.py" >/dev/null; do sleep 30; done
run soft_emo                                      # C
COMBO=$(python acc_analysis.py --combo)           # E
ts "combo components: [$COMBO]"
if [ -n "$COMBO" ] && [ "$(echo $COMBO | tr ',' '\n' | wc -l)" -ge 2 ]; then
  for s in 1 2 3 4 5; do AUSOFT2_COMBO=$COMBO python au_soft2.py --variant soft_combo --seed $s > $F/logs/acc_combo_s$s.log 2>&1 \
    && ts "trained combo $s" || ts "FAILED combo $s"; done
else ts "combo skipped (fewer than 2 helpful components)"; fi
run soft_diff                                     # F
run xattn_pre; run selfattn_pre                   # D, D0
python acc_analysis.py > $F/logs/acc_analysis.log 2>&1
ts "ACC_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
