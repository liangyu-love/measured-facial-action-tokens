"""Evidence-quality round summary. `--decide` prints the evidence variant to use for the diff2 re-run (or NONE)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import FULL, metrics

EVS = ["raw8", "raw8emo", "dense", "rel", "z"]


def evq(ev):
    return [json.load(open(p)) for p in glob.glob(os.path.join(FULL, "results_evq", f"{ev}_seed*.json"))]


if "--decide" in sys.argv:
    corr = {ev: np.mean([r["test_labelV"]["corr"] for r in evq(ev)]) for ev in EVS if evq(ev)}
    # choose the best on VALID MAE (not test), then require a clear test-Corr gain over raw8emo for the diff2 re-run
    vmae = {ev: np.mean([r["valid_mae"] for r in evq(ev)]) for ev in EVS if evq(ev)}
    best = min(vmae, key=vmae.get)
    ok = best != "raw8emo" and corr.get(best, 0) - corr.get("raw8emo", 0) >= 0.03
    print(best if ok else "NONE")
    sys.exit(0)


def load(d, v):
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(FULL, d, f"{v}_seed*.json"))}
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(FULL, d, f"{v}_seed*_test_pred.csv")))]
    return js, (pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(1)) if pr else None)


def boot(a, b, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[metric] - metrics(y, pb)[metric]
    d = np.array([metrics(y[s], pa[s])[metric] - metrics(y[s], pb[s])[metric] for s in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


L = ["### (1) evidence quality: visual-only model -> label_V (test; mean±std over 5 seeds)",
     "| evidence | valid MAE | test Corr | test Acc-2 | test MAE | Conflict Corr |", "|---|---|---|---|---|---|"]
for ev in EVS:
    rs = evq(ev)
    if rs:
        f = lambda k, sub="test_labelV": f"{np.mean([r[sub][k] for r in rs]):.4f}±{np.std([r[sub][k] for r in rs], ddof=1):.4f}"
        name = ev + (f" ({rs[0].get('dense_variant')})" if ev == "dense" else "")
        L.append(f"| {name} | {np.mean([r['valid_mae'] for r in rs]):.4f} | {f('corr')} | {f('acc2')} | {f('mae')} | {f('corr', 'test_labelV_conflict')} |")
base_js, base = load("results_acc", "soft_emo")
L += ["\n### (2) per-video normalisation fed to au_soft+emo (mean±std over seeds)", "| model | subset | Acc-2 | MAE | Corr |", "|---|---|---|---|---|"]
rows = [("au_soft+emo", ("results_acc", "soft_emo")), ("+ video-median (rel)", ("results_norm_rel", "soft_emo")),
        ("+ video z-score (z)", ("results_norm_z", "soft_emo"))]
for name, (d, v) in rows:
    js, _ = load(d, v)
    for sub in ("test", "test_conflict", "test_align"):
        if len(js) > 1:
            rs = list(js.values())
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            L.append(f"| {name} | {sub} | {f('acc2')} | {f('mae')} | {f('corr')} |")
L += ["\n| variant | subset | metric | diff vs au_soft+emo | t p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|"]
for name, (d, v) in rows[1:]:
    js, av = load(d, v)
    if av is None or len(js) < 2:
        continue
    seeds = sorted(set(js) & set(base_js))
    for sub in ("full", "conflict", "align"):
        key = "test" if sub == "full" else f"test_{sub}"
        a = av if sub == "full" else av[(av.category == "conflict") == (sub == "conflict")]
        for m in ("acc2", "mae", "corr"):
            tp = ttest_rel([js[s][key][m] for s in seeds], [base_js[s][key][m] for s in seeds]).pvalue
            obs, lo, hi, bp = boot(a, base, m)
            L.append(f"| {name} | {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
d2 = [json.load(open(p)) for p in glob.glob(os.path.join(FULL, "results_diff2_ev", "diff2_uni_seed*.json"))]
if d2:
    L += ["\n### (3) diff2_uni on the better evidence (mean over seeds)"]
    for sub in ("test", "test_conflict"):
        L.append(f"- {sub}: Acc-2 {np.mean([r[sub]['acc2'] for r in d2]):.4f}, MAE {np.mean([r[sub]['mae'] for r in d2]):.4f}, Corr {np.mean([r[sub]['corr'] for r in d2]):.4f}")
    dg = [r["diag"] for r in d2]
    L.append(f"- shuffle ΔCorr full {np.mean([x['shuffle_delta_full']['corr'] for x in dg]):+.4f}, Conflict {np.mean([x['shuffle_delta_conflict']['corr'] for x in dg]):+.4f}; "
             f"Spearman(‖z_v−z_t‖,|T−V|) {np.mean([x['spearman_gap_vs_trueTVgap'] for x in dg]):+.3f}; AUROC sign conflict {np.mean([x['auroc_gap_sign_conflict'] for x in dg]):.3f}")
txt = "\n".join(L)
open(os.path.join(FULL, "evq_results.md"), "w").write(txt)
print(txt)
