#!/usr/bin/env bash
# Evidence-attending ([EVI]) experiments; pauses the Qwen3-VL queue while running, resumes it at the end.
source ~/.venv/bin/activate
export HF_HUB_OFFLINE=1
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -qE "CLIP_ALL_DONE|CLIP_FAILED" $F/run_clip.log 2>/dev/null; do sleep 30; done
pkill -f "^bash scripts/run_qwen3.sh"; pkill -f "^python scale_desc.py"; pkill -f "^python gen_full.py"; pkill -f "^python sft_qwenvl.py"
ts "qwen3 queue paused"
for v in fa evi fa_evi fa_evi_reg; do for s in 1 2 3 4 5; do
  python evi_bert.py --variant $v --seed $s > $F/logs/evi_${v}_s$s.log 2>&1 && ts "trained $v $s" || ts "FAILED $v $s"; done; done
for v in fa_evi fa_evi_reg; do python evi_bert.py --variant $v --seed 1 --maps > $F/logs/evi_maps_$v.log 2>&1 && ts "maps $v done"; done
python sig_tests.py > $F/logs/sig_tests.log 2>&1 && ts "significance done"
python - <<'PY' > $F/evi_results.md
import glob, json, os, numpy as np
F = os.path.expanduser("~/work/msa-pilot/outputs/full")
def g(d, name):
    return [json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{name}_seed*.json"))]
rows = [("au_soft", g("results_ausoft", "soft")), ("fa", g("results_evi", "fa")), ("evi", g("results_evi", "evi")),
        ("fa_evi", g("results_evi", "fa_evi")), ("fa_evi_reg", g("results_evi", "fa_evi_reg"))]
for sub in ("test", "test_conflict", "test_align"):
    print(f"\n### {sub}\n| model | seeds | Acc-2 | F1 | MAE | Corr |\n|---|---|---|---|---|---|")
    for n, rs in rows:
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            print(f"| {n} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
print("\n### faithfulness (test, delete top/bottom/random 10% evidence tokens; mean over seeds)")
for n in ("evi", "fa_evi", "fa_evi_reg"):
    rs = [r["faithfulness"] for r in g("results_evi", n) if "faithfulness" in r]
    if not rs:
        continue
    print(f"\n{n}: k={rs[0]['k']} of {rs[0]['n_evidence_tokens']}; S entropy {np.mean([r['S_entropy_mean'] for r in rs]):.3f}; "
          f"S-vs-rollout spearman {np.mean([r['S_vs_rollout_spearman'] for r in rs]):.3f}; "
          f"top-10% mass {np.mean([r.get('S_top10_mass', np.nan) for r in rs]):.3f}; "
          f"gap top-vs-random {np.mean([r.get('gap_top_vs_random', np.nan) for r in rs]):+.4f}")
    print("| ranking | mean |Δpred| | ΔMAE | ΔMAE (Conflict) |\n|---|---|---|---|")
    for key in ("S_top", "rollout_top", "random", "S_bottom"):
        m = lambda q: f"{np.mean([r[key][q] for r in rs]):+.4f}±{np.std([r[key][q] for r in rs], ddof=1):.4f}"
        print(f"| {key} | {m('mean_abs_dpred')} | {m('dMAE')} | {m('dMAE_conflict')} |")
PY
ts "EVI_ALL_DONE"
cd ~/work/msa-pilot && (setsid nohup bash scripts/run_qwen3.sh > outputs/run_qwen3.log 2>&1 < /dev/null &)
ts "qwen3 queue resumed"
