#!/usr/bin/env bash
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
S=~/work/msa-pilot/outputs/pilot_scale
until grep -q GATE_ALL_DONE ../outputs/full/run_gate.log; do sleep 60; done
for m in 3B 32B; do
  until ! pgrep -f "[h]f download Qwen/Qwen2.5-VL-$m-Instruct" >/dev/null; do sleep 60; done
  echo "$(date +%T) generate $m"
  python scale_desc.py --size $m > $S/gen_$m.log 2>&1 || echo "$(date +%T) FAILED generate $m"
done
for m in 3B 7B 32B; do
  [ -s $S/$m/desc.jsonl ] || continue
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_DESC=$S/$m/desc.jsonl SCORE_OUT=$S/$m python score_v3.py > $S/score_$m.log 2>&1
  CUDA_VISIBLE_DEVICES= SCORE_VERSION=v3b SCORE_DESC=$S/$m/desc.jsonl SCORE_KEYS=../outputs/pilot_codex20/selected_clips.csv SCORE_OUT=$S/$m/sub20 python score_v3.py > $S/score_${m}_sub20.log 2>&1
  echo "$(date +%T) scored $m"
done
echo "SCALE_ALL_DONE $(date +%T)"
