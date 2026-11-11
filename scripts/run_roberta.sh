#!/usr/bin/env bash
# Encoder robustness: hfl/chinese-roberta-wwm-ext instead of bert-base-chinese (same setup, 5 seeds).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
E=~/work/msa-pilot/models/chinese-roberta-wwm-ext
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -q SEMAB_ALL_DONE $F/run_semab.log 2>/dev/null; do sleep 30; done
until [ -s $E/pytorch_model.bin ] && [ -s $E/config.json ] && ! pgrep -f "[h]f download hfl/chinese-roberta" >/dev/null; do sleep 30; done
ts "start roberta"
export ENCODER_DIR=$E RESULTS_DIR=results_roberta XGATE_RESULTS_DIR=results_roberta_xgate
for c in text deva_av tmpl_au; do for s in 1 2 3 4 5; do
  python train_bert.py --cond $c --seed $s > $F/logs/rob_${c}_s$s.log 2>&1 && ts "trained $c $s" || ts "FAILED $c $s"; done; done
for v in zh_concat xgate_noaux; do for s in 1 2 3 4 5; do
  python xgate_bert.py --variant $v --seed $s > $F/logs/rob_${v}_s$s.log 2>&1 && ts "trained $v $s" || ts "FAILED $v $s"; done; done
python - <<'PY' > $F/roberta_results.md
import glob, json, os
import numpy as np
F = os.path.expanduser("~/work/msa-pilot/outputs/full")
def load(d, key):
    out = {}
    for p in glob.glob(os.path.join(F, d, "*_seed*.json")):
        r = json.load(open(p)); out.setdefault(r[key], []).append(r)
    return out
bert = {**load("results", "cond"), **load("results_xgate", "variant")}
rob = {**load("results_roberta", "cond"), **load("results_roberta_xgate", "variant")}
for sub in ("test", "test_conflict", "test_align"):
    print(f"\n### {sub}\n| model | bert-base-chinese Acc-2 / MAE / Corr | chinese-roberta-wwm-ext Acc-2 / MAE / Corr |\n|---|---|---|")
    for c in ("text", "tmpl_au", "deva_av", "zh_concat", "xgate_noaux"):
        cell = lambda rs: " / ".join(f"{np.mean([x[sub][m] for x in rs]):.4f}±{np.std([x[sub][m] for x in rs], ddof=1):.4f}"
                                     for m in ("acc2", "mae", "corr")) if rs and len(rs) > 1 else "n/a"
        print(f"| {c} | {cell(bert.get(c, []))} | {cell(rob.get(c, []))} |")
PY
ts "ROBERTA_ALL_DONE"
