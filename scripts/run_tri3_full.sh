#!/usr/bin/env bash
# Full protocol for the post-freeze three-expert variant (user decision 11-08): experts for seeds 6-10 -> wait for the Qwen3-VL 8B
# descriptions (+ verification + vlm8b conditions) -> pause Qwen3 -> uniform tuning (valid only, same grid) for diff3_tri3_trust and its
# no-voice counterpart diff3_oof -> 10 test seeds -> significance -> resume Qwen3 (resumable; 4B SFT and 32B follow).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1 MSA_DATASET=sims2 MSA_DATA=outputs/full
cd ~/work/msa-pilot/scripts
R=~/work/msa-pilot/outputs; F=$R/full
ts() { echo "[$(date '+%F %T')] $*"; }
free_ok() { [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt ${1:-5000} ]; }
ts "start: experts for seeds 6-10"
for s in 6 7 8 9 10; do
  until free_ok 3000; do sleep 30; done; python evq.py --ev raw8emo --seed $s > $F/logs/evq_raw8emo_s$s.log 2>&1 || ts "FAILED evq $s"
  until free_ok 5000; do sleep 30; done; python texp.py --seed $s > $F/logs/texp_s$s.log 2>&1 || ts "FAILED texp $s"
  until free_ok 5000; do sleep 30; done; python oof_experts.py --seed $s > $F/logs/oof_experts_s$s.log 2>&1 && ts "oof experts $s" || ts "FAILED oof $s"
done
until free_ok 3000; do sleep 30; done
python audio_expert_hubert.py 6,7,8,9,10 > $F/logs/audio_expert_hubert_6_10.log 2>&1; ts "audio experts 6-10: $(grep MEAN $F/logs/audio_expert_hubert_6_10.log)"
ts "experts: oof $(ls $F/expert_oof/seed*.csv | grep -v reg | wc -l), audio $(ls $F/expert_oof/audio_hubert_seed*.csv | wc -l)"
until grep -qE "SFT train|gen 4B zs|QWEN3_ALL_DONE" $R/run_qwen3.log 2>/dev/null; do sleep 60; done
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python sft_qwenvl.py"; pkill -f "^python gen_full.py"; pkill -f "^python scale_desc.py"; sleep 10
ts "qwen3 paused after 8B descriptions: $(grep -E 'verified 8B|trained vlm8b' $R/run_qwen3.log | tail -2 | tr '\n' ' ')"
[ -f $F/results_bert2/trust_choice.json ] && mv $F/results_bert2/trust_choice.json $F/results_bert2/trust_choice.json.off  # not part of this round
python tune.py stage1 --dataset sims2 --workers 3 --dense_variant crop8 > $F/logs/tune_stage1_tri3.log 2>&1 && ts "stage1 (new methods only)"
python tune.py stage2 --dataset sims2 --workers 3 --dense_variant crop8 > $F/logs/tune_stage2_tri3.log 2>&1 && ts "stage2 seeds 1-5"
python tune.py stage2 --dataset sims2 --workers 3 --dense_variant crop8 --seed_from 6 --seed_to 10 > $F/logs/tune_stage2_tri3_s6_10.log 2>&1 && ts "stage2 seeds 6-10"
python tune.py summary --dataset sims2 --dense_variant crop8 > /dev/null 2>&1
python peek_final.py > $F/logs/peek_final_tri3.log 2>&1; python audit.py > $F/logs/audit_tri3full.log 2>&1
ts "TRI3_FULL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh >> outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
