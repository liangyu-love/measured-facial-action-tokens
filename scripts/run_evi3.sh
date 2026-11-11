#!/usr/bin/env bash
# EVI round 3 (fa2 / post-hoc / 2-stage / visual-label supervision). Pauses and then resumes the Qwen3-VL queue.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
R=$F/results_evi
ts() { echo "[$(date '+%F %T')] $*"; }
resume() { cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &); ts "qwen3 resumed"; }
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
sleep 5; ts "qwen3 paused"
# smoke (seed 0, 1 epoch) for all four modes
ok=1
python evi3.py --variant fa2 --seed 0 --epochs 1 > $F/logs/evi3_smoke.log 2>&1 || ok=0
[ $ok = 1 ] && { python evi3.py --variant fa2 --seed 0 --posthoc >> $F/logs/evi3_smoke.log 2>&1 || ok=0; }
[ $ok = 1 ] && { python evi3.py --variant fa_evi_2stage --seed 0 --epochs 1 >> $F/logs/evi3_smoke.log 2>&1 || ok=0; }
[ $ok = 1 ] && { python evi3.py --variant fa_evi_vsup --seed 0 --epochs 1 >> $F/logs/evi3_smoke.log 2>&1 || ok=0; }
rm -f $R/*_seed0* ; rm -rf $F/evi_maps/fa2_posthoc_L12 $F/evi_maps/fa2_posthoc_L8 $F/evi_maps/fa_evi_2stage $F/evi_maps/fa_evi_vsup
if [ $ok = 0 ]; then ts "SMOKE FAILED (see logs/evi3_smoke.log)"; resume; ts "EVI3_ABORTED"; exit 1; fi
ts "smoke ok"
for s in 1 2 3 4 5; do python evi3.py --variant fa2 --seed $s > $F/logs/evi3_fa2_s$s.log 2>&1 && ts "trained fa2 $s" || ts "FAILED fa2 $s"; done
for s in 1 2 3 4 5; do python evi3.py --variant fa2 --seed $s --posthoc > $F/logs/evi3_post_s$s.log 2>&1 && ts "posthoc $s" || ts "FAILED posthoc $s"; done
for s in 1 2 3 4 5; do python evi3.py --variant fa_evi_2stage --seed $s > $F/logs/evi3_2stage_s$s.log 2>&1 && ts "trained 2stage $s" || ts "FAILED 2stage $s"; done
for s in 1 2 3 4 5; do python evi3.py --variant fa_evi_vsup --seed $s > $F/logs/evi3_vsup_s$s.log 2>&1 && ts "trained vsup $s" || ts "FAILED vsup $s"; done
python sig_tests.py > $F/logs/sig_tests.log 2>&1 && ts "significance done"
python - <<'PY' > $F/evi3_results.md
import glob, json, os, numpy as np
F = os.path.expanduser("~/work/msa-pilot/outputs/full")
g = lambda d, n: [json.load(open(p)) for p in sorted(glob.glob(os.path.join(F, d, f"{n}_seed*.json")))]
rows = [("au_soft", g("results_ausoft", "soft")), ("fa_evi2", g("results_evi", "fa_evi2")), ("fa2", g("results_evi", "fa2")),
        ("fa_evi_2stage", g("results_evi", "fa_evi_2stage")), ("fa_evi_vsup", g("results_evi", "fa_evi_vsup"))]
for sub in ("test", "test_conflict", "test_align"):
    print(f"\n### {sub}\n| model | seeds | Acc-2 | F1 | MAE | Corr |\n|---|---|---|---|---|---|")
    for n, rs in rows:
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            print(f"| {n} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
print("\n### map quality / faithfulness (mean over seeds; delete 10% of 160 evidence tokens; |Δpred|)")
print("| map | S entropy (uniform=5.075) | top-10% mass | S-vs-rollout ρ | S_top | rollout_top | random | S_bottom | gap top−random |")
print("|---|---|---|---|---|---|---|---|---|")
def row(name, fs):
    m = lambda k: np.mean([f[k] for f in fs])
    d = lambda k: np.mean([f[k]["mean_abs_dpred"] for f in fs])
    print(f"| {name} | {m('S_entropy_mean'):.3f} | {m('S_top10_mass'):.3f} | {m('S_vs_rollout_spearman'):+.3f} | {d('S_top'):.4f} | "
          f"{d('rollout_top'):.4f} | {d('random'):.4f} | {d('S_bottom'):.4f} | {d('S_top') - d('random'):+.4f} |")
for n in ("fa_evi2", "fa_evi2_sup", "fa_evi_2stage", "fa_evi_vsup"):
    fs = [r["faithfulness"] for r in g("results_evi", n) if "faithfulness" in r]
    if fs: row(n, fs)
post = g("results_evi", "fa2_posthoc")
for L in ("L12", "L8"):
    fs = [r["faithfulness"][L] for r in post if "faithfulness" in r]
    if fs: row(f"fa2 post-hoc CLS {L}", fs)
PY
ts "EVI3_ALL_DONE"
resume
