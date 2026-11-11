#!/usr/bin/env bash
# One-shot EVI fix (fa_evi2, fa_evi2_sup): runs after the soft-gate queue, pauses/resumes the Qwen3-VL queue.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -q GSOFT_ALL_DONE $F/run_gsoft.log 2>/dev/null; do sleep 20; done
sleep 5; pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 paused"
for v in fa_evi2 fa_evi2_sup; do for s in 1 2 3 4 5; do
  python evi_bert.py --variant $v --seed $s > $F/logs/evi_${v}_s$s.log 2>&1 && ts "trained $v $s" || ts "FAILED $v $s"; done
  python evi_bert.py --variant $v --seed 1 --maps > $F/logs/evi_maps_$v.log 2>&1 && ts "maps $v done"; done
python sig_tests.py > $F/logs/sig_tests.log 2>&1 && ts "significance done"
python - <<'PY' > $F/evi2_results.md
import glob, json, os, numpy as np
F = os.path.expanduser("~/work/msa-pilot/outputs/full")
g = lambda d, n: [json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{n}_seed*.json"))]
rows = [("au_soft", g("results_ausoft", "soft")), ("fa_evi (v1)", g("results_evi", "fa_evi")),
        ("fa_evi2", g("results_evi", "fa_evi2")), ("fa_evi2_sup", g("results_evi", "fa_evi2_sup"))]
for sub in ("test", "test_conflict", "test_align"):
    print(f"\n### {sub}\n| model | seeds | Acc-2 | F1 | MAE | Corr |\n|---|---|---|---|---|---|")
    for n, rs in rows:
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            print(f"| {n} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
print("\n### map quality / faithfulness (mean over seeds; delete 10% of 160 evidence tokens)")
print("| model | S entropy (uniform=5.075) | top-10% mass (uniform=0.10) | S-vs-rollout ρ | |Δpred| S_top | rollout_top | random | S_bottom | gap top−random |")
print("|---|---|---|---|---|---|---|---|---|")
for n in ("fa_evi", "fa_evi_reg", "fa_evi2", "fa_evi2_sup"):
    rs = [r["faithfulness"] for r in g("results_evi", n) if "faithfulness" in r]
    if not rs:
        continue
    m = lambda k: np.mean([r[k] for r in rs]) if k in rs[0] else np.nan
    d = lambda k: np.mean([r[k]["mean_abs_dpred"] for r in rs])
    print(f"| {n} | {m('S_entropy_mean'):.3f} | {m('S_top10_mass'):.3f} | {m('S_vs_rollout_spearman'):+.3f} | "
          f"{d('S_top'):.4f} | {d('rollout_top'):.4f} | {d('random'):.4f} | {d('S_bottom'):.4f} | {d('S_top') - d('random'):+.4f} |")
PY
ts "EVI2_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 resumed"
