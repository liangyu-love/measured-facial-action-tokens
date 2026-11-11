#!/usr/bin/env bash
# Core-method experiments (ahead of the Qwen3-VL queue). Resumable (finished seeds are skipped).
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
run() { local v=$1; shift; for s in 1 2 3 4 5; do env "$@" python au_soft.py --variant $v --seed $s > $F/logs/core_${v}_${RES:-base}_s$s.log 2>&1 \
  && ts "trained $v ${RES:-base} $s" || ts "FAILED $v ${RES:-base} $s"; done; }

# 1) au_soft with RoBERTa
RES=roberta run soft ENCODER_DIR=$HOME/work/msa-pilot/models/chinese-roberta-wwm-ext AUSOFT_RESULTS_DIR=results_roberta_ausoft
# 2) salient-only AUs, 3) + head/gaze, 6) MAG-BERT-style
for v in soft_salient soft_hg mag; do RES= run $v AUSOFT_RESULTS_DIR=results_ausoft; done
# 3b) + audio token (needs audio88)
until [ -s $F/audio_egemaps88.csv ] && ! pgrep -f "^python audio88.py" >/dev/null; do sleep 30; done
RES= run soft_hga AUSOFT_RESULTS_DIR=results_ausoft
# 5) generic visual features (CLIP ViT-B/16)
[ -s $F/clip_b16_frames.npy ] || python clip_feats.py > $F/logs/clip_feats.log 2>&1 || ts "FAILED clip feats"
RES= run soft_clip AUSOFT_RESULTS_DIR=results_ausoft
# 4) significance
python sig_tests.py > $F/logs/sig_tests.log 2>&1 && ts "significance done"
# summary table
python - <<'PY' > $F/core_results.md
import glob, json, os, numpy as np
F = os.path.expanduser("~/work/msa-pilot/outputs/full")
def g(d, key, name):
    return [json.load(open(p)) for p in glob.glob(os.path.join(F, d, "*_seed*.json")) if json.load(open(p))[key] == name]
rows = [("text", g("results", "cond", "text")), ("tmpl_au", g("results", "cond", "tmpl_au")), ("deva_av", g("results", "cond", "deva_av")),
        ("zh_code_noid", g("results_xgate", "variant", "zh_code_noid")), ("au_late", g("results_ausoft", "variant", "late")),
        ("mag", g("results_ausoft", "variant", "mag")), ("soft_clip", g("results_ausoft", "variant", "soft_clip")),
        ("soft_salient", g("results_ausoft", "variant", "soft_salient")), ("au_soft", g("results_ausoft", "variant", "soft")),
        ("soft_hg", g("results_ausoft", "variant", "soft_hg")), ("soft_hga", g("results_ausoft", "variant", "soft_hga")),
        ("au_soft[roberta]", g("results_roberta_ausoft", "variant", "soft")),
        ("deva_av[roberta]", g("results_roberta", "cond", "deva_av")), ("tmpl_au[roberta]", g("results_roberta", "cond", "tmpl_au"))]
for sub in ("test", "test_conflict", "test_align"):
    print(f"\n### {sub}\n| model | seeds | Acc-2 | F1 | MAE | Corr |\n|---|---|---|---|---|---|")
    for n, rs in rows:
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            print(f"| {n} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
PY
ts "CORE_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 queue resumed"
