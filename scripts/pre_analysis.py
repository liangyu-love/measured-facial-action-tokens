"""Pretrained-evidence round summary for the current MSA_DATA / MSA_DATASET -> <FULL>/pre_results.md"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import DATASET, FULL, metrics

MS = ["acc2", "acc2_non0", "acc7", "mae", "corr"] if DATASET in ("mosi", "mosei") else ["acc2", "mae", "corr"]


def load(d, v):
    js = {}
    for p in glob.glob(os.path.join(FULL, d, f"{v}_seed*.json")):
        r = json.load(open(p)); js[r["seed"]] = r
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(FULL, d, f"{v}_seed*_test_pred.csv")))]
    av = pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(1)) if pr else None
    return js, av


def boot(a, b, m, n=2000):
    rng = np.random.default_rng(0); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[m] - metrics(y, pb)[m]
    d = np.array([metrics(y[s], pa[s])[m] - metrics(y[s], pb[s])[m] for s in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


rows = [("au_soft+emo", "results_acc", "soft_emo"), ("diff2_uni (raw AU)", "results_diff2", "diff2_uni"),
        ("au_soft_pre", "results_pre", "au_soft_pre"), ("diff2_pre", "results_pre", "diff2_pre")]
subs = ["test"] + (["test_conflict", "test_align"] if DATASET == "sims2" else [])
L = [f"## Pretrained (SIMS label_V) frozen evidence encoder — {DATASET} (mean±std over seeds)",
     "| model | subset | n seeds | " + " | ".join(MS) + " |", "|---|---|---|" + "---|" * len(MS)]
data = {}
for name, d, v in rows:
    js, av = load(d, v)
    if not js:
        continue
    data[name] = (js, av)
    for sub in subs:
        rs = list(js.values())
        cell = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1) if len(rs) > 1 else 0:.4f}"
        L.append(f"| {name} | {sub} | {len(rs)} | " + " | ".join(cell(m) for m in MS) + " |")
ev = next((r["ev"] for n in ("au_soft_pre", "diff2_pre") if n in data for r in data[n][0].values()), None)
L.append(f"\nencoder evidence variant: **{ev}**")
L += ["\n| variant | vs | subset | metric | diff | t p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|---|"]
for name, ref in (("au_soft_pre", "au_soft+emo"), ("diff2_pre", "au_soft+emo"), ("diff2_pre", "diff2_uni (raw AU)")):
    if name not in data or ref not in data or data[name][1] is None:
        continue
    (ja, aa), (jb, ab) = data[name], data[ref]
    seeds = sorted(set(ja) & set(jb))
    for sub in [s.replace("test_", "") if s != "test" else "full" for s in subs]:
        key = "test" if sub == "full" else f"test_{sub}"
        a = aa if sub == "full" else aa[(aa.category == "conflict") == (sub == "conflict")]
        for m in ("acc2", "mae", "corr"):
            tp = ttest_rel([ja[s][key][m] for s in seeds], [jb[s][key][m] for s in seeds]).pvalue if len(seeds) > 1 else float("nan")
            o, lo, hi, bp = boot(a, ab, m)
            L.append(f"| {name} | {ref} | {sub} | {m} | {o:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
L.append("\n### diagnostics (mean over seeds)")
for name in ("au_soft_pre", "diff2_pre"):
    if name in data:
        dg = [r["diag"] for r in data[name][0].values()]
        keys = sorted(set().union(*dg))
        for k in keys:
            vals = [x[k] for x in dg if k in x]
            s = ", ".join(f"{m} {np.mean([v[m] for v in vals]):+.4f}" for m in vals[0]) if isinstance(vals[0], dict) else f"{np.mean(vals):+.4f}"
            L.append(f"- {name} · {k}: {s}")
txt = "\n".join(L)
open(os.path.join(FULL, "pre_results.md"), "w").write(txt)
print(txt)
