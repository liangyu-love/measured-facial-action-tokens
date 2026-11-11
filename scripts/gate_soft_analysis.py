"""Soft-token gate variants vs au_soft: metrics, gate-value analysis, significance (per-seed paired t + 2k bootstrap)."""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

from train_bert import metrics

F = os.path.expanduser("~/work/msa-pilot/outputs/full")


def load(d, v):
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{v}_seed*.json"))}
    preds = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(F, d, f"{v}_seed*_test_pred.csv")))]
    return js, preds


def boot(a, b, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[metric] - metrics(y, pb)[metric]
    d = np.array([metrics(y[s], pa[s])[metric] - metrics(y[s], pb[s])[metric]
                  for s in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / n)


base_js, base_pr = load("results_ausoft", "soft")
base = base_pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in base_pr], axis=1).mean(1))
lines = []
for sub in ("test", "test_conflict", "test_align"):
    lines += [f"\n### {sub}", "| model | seeds | Acc-2 | F1 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for name, (d, v) in (("au_soft", ("results_ausoft", "soft")), ("soft_gate", ("results_softgate", "soft_gate")),
                         ("soft_gate_aux", ("results_softgate", "soft_gate_aux")), ("soft_tokgate", ("results_softgate", "soft_tokgate"))):
        js, _ = load(d, v)
        rs = list(js.values())
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            lines.append(f"| {name} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
lines += ["\n### gate analysis (test; mean over seeds)",
          "| variant | g mean | g std (across clips) | g Conflict | g Align | AUROC g→TV-conflict | AUROC (1-g)→TV-conflict | AUROC g→Conflict category |",
          "|---|---|---|---|---|---|---|---|"]
sig = ["\n### significance vs au_soft (diff = variant − au_soft; t-test per-seed paired, bootstrap 2k on seed-averaged preds)",
       "| variant | subset | metric | diff | t-test p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|"]
for v in ("soft_gate", "soft_gate_aux", "soft_tokgate"):
    js, preds = load("results_softgate", v)
    if not preds:
        continue
    rows = []
    for p in preds:
        tv = (np.sign(p.label_T) != np.sign(p.label_V)).astype(int)
        rows.append([p.gate.mean(), p.gate.std(), p.gate[p.category == "conflict"].mean(), p.gate[p.category != "conflict"].mean(),
                     roc_auc_score(tv, p.gate), roc_auc_score(tv, 1 - p.gate),
                     roc_auc_score((p.category == "conflict").astype(int), p.gate)])
    r = np.mean(rows, 0)
    lines.append(f"| {v} | " + " | ".join(f"{x:.3f}" for x in r) + " |")
    pv = preds[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in preds], axis=1).mean(1))
    seeds = sorted(set(js) & set(base_js))
    for sub in ("full", "conflict", "align"):
        key = "test" if sub == "full" else f"test_{sub}"
        a = pv if sub == "full" else pv[(pv.category == "conflict") == (sub == "conflict")]
        for m in ("acc2", "mae", "corr"):
            tp = ttest_rel([js[s][key][m] for s in seeds], [base_js[s][key][m] for s in seeds]).pvalue
            obs, lo, hi, bp = boot(a, base, m)
            sig.append(f"| {v} | {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
txt = "\n".join(lines + sig)
open(os.path.join(F, "softgate_results.md"), "w").write(txt)
print(txt)
