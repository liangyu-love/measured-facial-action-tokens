#!/usr/bin/env bash
# Evidence-quality round (after the densification re-run, before MOSEI GPU stage and tuning). Qwen3-VL stays paused.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -qE "DENSE2_ALL_DONE|DENSE2_ABORTED" $F/run_dense2.log 2>/dev/null; do sleep 30; done
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"
ts "start (dense2: $(tail -1 $F/run_dense2.log | cut -c1-60))"
# (1) visual-only yardstick: 5 evidence variants x 5 seeds (tiny models, 5 in parallel)
for ev in raw8 raw8emo dense rel z; do
  printf "%s\n" 1 2 3 4 5 | xargs -P 5 -I{} bash -c "python evq.py --ev $ev --seed {} > $F/logs/evq_${ev}_s{}.log 2>&1 || echo FAILED $ev {}"
  ts "evq $ev done"
done
# (2) per-video normalisation into au_soft+emo (3 seeds in parallel)
for nm in rel z; do
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "MSA_NORM=$nm AUSOFT2_RESULTS_DIR=results_norm_$nm python au_soft2.py --variant soft_emo --seed {} > $F/logs/norm_${nm}_s{}.log 2>&1 || echo FAILED norm $nm {}"
  ts "norm $nm done"
done
# (3) conditional diff2_uni on the better evidence
EV=$(python evq_analysis.py --decide)
ts "decide: [$EV]"
if [ "$EV" = "rel" ] || [ "$EV" = "z" ] || [ "$EV" = "raw8emo" ]; then
  NORMV=""; [ "$EV" = "rel" ] || [ "$EV" = "z" ] && NORMV=$EV
  printf "%s\n" 1 2 3 4 5 | xargs -P 3 -I{} bash -c "MSA_NORM=$NORMV DIFF2_RESULTS_DIR=results_diff2_ev python diff2.py --variant diff2_uni --emo --seed {} > $F/logs/diff2ev_s{}.log 2>&1 || echo FAILED diff2ev {}"
  ts "diff2_uni on $EV done"
elif [ "$EV" = "dense" ]; then ts "diff2 re-run skipped: densified evidence is not wired into diff2 (would need a separate implementation)"
else ts "diff2 re-run skipped: no evidence variant clearly improved visual-only label_V Corr"; fi
python evq_analysis.py > $F/logs/evq_analysis.log 2>&1
ts "EVQ_ALL_DONE"
