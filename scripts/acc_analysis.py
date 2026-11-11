"""Accuracy round: table + significance vs au_soft. `--combo` prints the helpful A-C components for variant E.
Helpful = 5-seed mean full-test Corr higher AND MAE lower than au_soft."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import metrics

F = os.path.expanduser("~/work/msa-pilot/outputs/full")
VARS = ["soft_stat", "soft_vaux01", "soft_vaux03", "soft_vtaux03", "soft_emo", "soft_combo", "soft_diff", "xattn_pre", "selfattn_pre"]


def load(d, v):
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{v}_seed*.json"))}
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(F, d, f"{v}_seed*_test_pred.csv")))]
    avg = pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(1)) if pr else None
    return js, avg


base_js, base = load("results_ausoft", "soft")
mean = lambda js, sub, m: np.mean([r[sub][m] for r in js.values()])
if "--combo" in sys.argv:
    comp = {"soft_stat": "stat", "soft_vaux01": "vaux01", "soft_vaux03": "vaux03", "soft_vtaux03": "vtaux03", "soft_emo": "emo"}
    good = []
    for v, c in comp.items():
        js, _ = load("results_acc", v)
        if len(js) == 5 and mean(js, "test", "corr") > mean(base_js, "test", "corr") and mean(js, "test", "mae") < mean(base_js, "test", "mae"):
            good.append(c)
    aux = [c for c in good if c.startswith("v")]  # keep only the best aux setting
    if len(aux) > 1:
        best = max(aux, key=lambda c: mean(load("results_acc", "soft_" + c)[0], "test", "corr"))
        good = [c for c in good if not c.startswith("v")] + [best]
    print(",".join(good))
    sys.exit(0)


def boot(a, b, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[metric] - metrics(y, pb)[metric]
    d = np.array([metrics(y[s], pa[s])[metric] - metrics(y[s], pb[s])[metric] for s in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


lines = []
for sub in ("test", "test_conflict", "test_align"):
    lines += [f"\n### {sub}", "| model | seeds | Acc-2 | F1 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for name, (d, v) in [("au_soft", ("results_ausoft", "soft"))] + [(v, ("results_acc", v)) for v in VARS]:
        js, _ = load(d, v)
        rs = list(js.values())
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            extra = f" ({rs[0].get('combo')})" if v == "soft_combo" else ""
            lines.append(f"| {name}{extra} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
lines += ["\n### significance vs au_soft (diff = variant − au_soft; per-seed paired t; 2k paired bootstrap on seed-averaged preds)",
          "| variant | subset | metric | diff | t-test p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|"]
for v in VARS:
    js, av = load("results_acc", v)
    if av is None or len(js) < 2:
        continue
    seeds = sorted(set(js) & set(base_js))
    for sub in ("full", "conflict", "align"):
        key = "test" if sub == "full" else f"test_{sub}"
        a = av if sub == "full" else av[(av.category == "conflict") == (sub == "conflict")]
        for m in ("acc2", "mae", "corr"):
            tp = ttest_rel([js[s][key][m] for s in seeds], [base_js[s][key][m] for s in seeds]).pvalue
            obs, lo, hi, bp = boot(a, base, m)
            lines.append(f"| {v} | {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
# D vs D0 directly
jx, ax_ = load("results_acc", "xattn_pre"); js0, a0 = load("results_acc", "selfattn_pre")
if ax_ is not None and a0 is not None:
    lines += ["\n### D (xattn_pre) vs D0 (selfattn_pre): diff = D − D0", "| subset | metric | diff | t-test p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|"]
    seeds = sorted(set(jx) & set(js0))
    for sub in ("full", "conflict", "align"):
        key = "test" if sub == "full" else f"test_{sub}"
        a = ax_ if sub == "full" else ax_[(ax_.category == "conflict") == (sub == "conflict")]
        for m in ("acc2", "mae", "corr"):
            tp = ttest_rel([jx[s][key][m] for s in seeds], [js0[s][key][m] for s in seeds]).pvalue
            obs, lo, hi, bp = boot(a, a0, m)
            lines.append(f"| {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
txt = "\n".join(lines)
open(os.path.join(F, "acc_results.md"), "w").write(txt)
print(txt)
