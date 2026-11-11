"""Significance: au_soft vs baselines on the CH-SIMS v2 test split (full / Conflict / Align).
 (1) per-seed paired t-test (seed i of method A vs seed i of method B) on Acc-2, MAE, Corr
 (2) paired bootstrap over test clips (2k resamples) on the seed-averaged prediction: diff, 95% CI, two-sided p
Output: outputs/full/significance.md
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import metrics

F = os.path.expanduser("~/work/msa-pilot/outputs/full")
SRC = {"au_soft": ("results_ausoft", "soft"), "text": ("results", "text"), "tmpl_au": ("results", "tmpl_au"),
       "deva_av": ("results", "deva_av"), "zh_code_noid": ("results_xgate", "zh_code_noid"),
       "mag": ("results_ausoft", "mag"), "soft_clip": ("results_ausoft", "soft_clip"),
       "soft_salient": ("results_ausoft", "soft_salient"),
       "fa": ("results_evi", "fa"), "evi": ("results_evi", "evi"), "fa_evi": ("results_evi", "fa_evi"),
       "fa_evi2": ("results_evi", "fa_evi2"), "fa_evi2_sup": ("results_evi", "fa_evi2_sup"),
       "fa2": ("results_evi", "fa2"), "fa_evi_2stage": ("results_evi", "fa_evi_2stage"), "fa_evi_vsup": ("results_evi", "fa_evi_vsup")}


def load(name):
    d, v = SRC[name]
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{v}_seed*.json"))}
    preds = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(F, d, f"{v}_seed*_test_pred.csv")))]
    if not preds:
        return js, None
    P = pd.concat([p.pred for p in preds], axis=1).mean(1)
    base = preds[0][["label", "category"]]
    return js, base.assign(pred=P)


def boot(a, b, sub, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    da = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(da, pa)[metric] - metrics(da, pb)[metric]
    diffs = np.empty(n)
    for k in range(n):
        s = rng.choice(idx, len(idx))
        diffs[k] = metrics(da[s], pa[s])[metric] - metrics(da[s], pb[s])[metric]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, max(p, 1 / n)


A_js, A_pred = load("au_soft")
lines = ["# au_soft vs baselines (test). t-test: per-seed paired; bootstrap: 2k resamples of test clips, seed-averaged predictions",
         "| baseline | subset | metric | diff (au_soft − base) | t-test p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|"]
for base in [k for k in SRC if k != "au_soft"]:
    B_js, B_pred = load(base)
    if B_pred is None or not B_js:
        continue
    seeds = sorted(set(A_js) & set(B_js))
    for sub, mask in (("full", None), ("conflict", "conflict"), ("align", "align")):
        key = "test" if sub == "full" else f"test_{sub}"
        a = A_pred if mask is None else A_pred[(A_pred.category == "conflict") == (mask == "conflict")]
        for m in ("acc2", "mae", "corr"):
            xa = [A_js[s][key][m] for s in seeds]; xb = [B_js[s][key][m] for s in seeds]
            tp = ttest_rel(xa, xb).pvalue if len(seeds) > 1 else float("nan")
            obs, lo, hi, bp = boot(a, B_pred, sub, m)
            lines.append(f"| {base} | {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
txt = "\n".join(lines)
open(os.path.join(F, "significance.md"), "w").write(txt)
print(txt)
