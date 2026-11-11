"""Significance on the tuned runs (all available seeds, e.g. 10): per-seed paired t-test + 2k paired bootstrap on
seed-averaged test predictions. Pairs: every method vs au_soft_emo, and diff2_pre vs every method.
Usage: MSA_DATASET=sims2 MSA_DATA=outputs/full python tune_sig.py   -> <data>/tuned_significance.md"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import DATASET, FULL, metrics

T = os.path.join(FULL, "tuned")
MS = ["acc2", "mae", "corr"] + (["acc2_non0", "acc7"] if DATASET in ("mosi", "mosei") else [])
SUBS = ["full", "conflict", "align"] if DATASET == "sims2" else ["full"]


def load(m):
    js, pr = {}, {}
    for p in glob.glob(os.path.join(T, m, "*_seed*.json")):
        r = json.load(open(p)); s = int(re.search(r"_seed(\d+)\.json$", p).group(1))
        if "test" in r:
            js[s] = r
            q = p.replace(".json", "_test_pred.csv")
            if os.path.exists(q):
                pr[s] = pd.read_csv(q).set_index("key")
    if not pr:
        return js, None
    base = next(iter(pr.values()))
    return js, base[["label", "category"]].assign(pred=pd.concat([p.pred for p in pr.values()], axis=1).mean(axis=1))


def sub(df, s):
    if s == "full":
        return df
    return df[df.category == "conflict"] if s == "conflict" else df[df.category.isin(["equal", "spne"])]


def boot(a, b, m, n=2000):
    rng = np.random.default_rng(0); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[m] - metrics(y, pb)[m]
    d = np.array([metrics(y[i], pa[i])[m] - metrics(y[i], pb[i])[m] for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


ms = sorted(d for d in os.listdir(T) if os.path.isdir(os.path.join(T, d)))
data = {m: load(m) for m in ms}
pairs = [(m, "au_soft_emo") for m in ms if m != "au_soft_emo"] + [("diff2_pre", m) for m in ms if m not in ("diff2_pre", "au_soft_emo")]
L = [f"# tuned significance ({DATASET}); seeds per method: " + ", ".join(f"{m}={len(data[m][0])}" for m in ms),
     "| variant | vs | subset | metric | diff | t p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|---|"]
for a_, b_ in pairs:
    if a_ not in data or b_ not in data or data[a_][1] is None or data[b_][1] is None:
        continue
    (ja, pa), (jb, pb) = data[a_], data[b_]
    seeds = sorted(set(ja) & set(jb))
    for s in SUBS:
        key = "test" if s == "full" else f"test_{s}"
        for m in MS:
            tp = ttest_rel([ja[k][key][m] for k in seeds], [jb[k][key][m] for k in seeds]).pvalue if len(seeds) > 1 else float("nan")
            o, lo, hi, bp = boot(sub(pa, s), pb, m)
            L.append(f"| {a_} | {b_} | {s} | {m} | {o:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
txt = "\n".join(L)
open(os.path.join(FULL, "tuned_significance.md"), "w").write(txt)
print(txt)
