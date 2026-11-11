"""Tuned 10-seed summary (SIMS v2, MOSI) + strict subset, and DeBERTa MOSI (5 seeds, default recipe). CPU only."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

R = os.path.expanduser("~/work/msa-pilot/outputs")


def mets(y, p, ds):
    if ds == "sims2":
        p = np.clip(p, -1, 1); return {"acc2": ((y > 0) == (p > 0)).mean(), "mae": np.abs(y - p).mean(), "corr": np.corrcoef(y, p)[0, 1]}
    p = np.clip(p, -3, 3); nz = y != 0
    return {"acc2": ((y >= 0) == (p >= 0)).mean(), "acc2_non0": ((y[nz] > 0) == (p[nz] > 0)).mean(), "acc7": (np.round(np.clip(y, -3, 3)) == np.round(p)).mean(),
            "mae": np.abs(y - p).mean(), "corr": np.corrcoef(y, p)[0, 1]}


def preds(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*_seed*_test_pred.csv")):
        out[int(re.search(r"seed(\d+)_test", p).group(1))] = pd.read_csv(p).set_index("key")
    return out


for ds, sub in (("sims2", "full"), ("mosi", "mosi")):
    c = pd.read_csv(os.path.join(R, sub, "clips.csv")).set_index("key"); te = c[c.split == "test"]
    S = {"full": te.index}
    if ds == "sims2":
        S["conflict"] = te.index[te.category == "conflict"]; S["strict"] = te.index[(np.sign(te.label_T) * np.sign(te.label_V)) < 0]
    T = os.path.join(R, sub, "tuned"); allp = {}
    print(f"\n### {ds} tuned (valid-selected config), all available seeds; per-seed mean")
    for m in sorted(os.listdir(T)):
        pr = preds(os.path.join(T, m))
        if not pr:
            continue
        allp[m] = pr
        cells = []
        for sn, ix in S.items():
            ms = [mets(te.label.loc[ix].values, p.pred.loc[ix].values, ds) for p in pr.values()]
            ks = ("acc2", "mae", "corr") if ds == "sims2" else ("acc2", "acc2_non0", "acc7", "mae", "corr")
            cells.append(sn + ": " + " ".join(f"{k} {np.mean([x[k] for x in ms]):.4f}" for k in ks))
        print(f"{m:14s} n={len(pr)} | " + " | ".join(cells))
    ref = "au_soft_emo"
    for m in allp:
        if m == ref or ref not in allp:
            continue
        seeds = sorted(set(allp[m]) & set(allp[ref])); line = []
        for sn, ix in S.items():
            for k in ("acc2", "corr", "mae"):
                a = [mets(te.label.loc[ix].values, allp[m][s].pred.loc[ix].values, ds)[k] for s in seeds]
                b = [mets(te.label.loc[ix].values, allp[ref][s].pred.loc[ix].values, ds)[k] for s in seeds]
                line.append(f"{sn}.{k} {np.mean(a) - np.mean(b):+.4f} (p={ttest_rel(a, b).pvalue:.3f})")
        print(f"  {m} vs {ref} ({len(seeds)} seeds): " + "; ".join(line))
    if "text" in allp and ref in allp:
        seeds = sorted(set(allp["text"]) & set(allp[ref]))
        for sn, ix in S.items():
            a = [mets(te.label.loc[ix].values, allp[ref][s].pred.loc[ix].values, ds)["acc2"] for s in seeds]
            b = [mets(te.label.loc[ix].values, allp["text"][s].pred.loc[ix].values, ds)["acc2"] for s in seeds]
            print(f"  au_soft_emo vs text {sn}.acc2 {np.mean(a) - np.mean(b):+.4f} (p={ttest_rel(a, b).pvalue:.3g})")
c = pd.read_csv(os.path.join(R, "mosi", "clips.csv")).set_index("key"); te = c[c.split == "test"]
print("\n### MOSI DeBERTa-v3-base (default recipe) vs BERT-base (default recipe)")
for nm, dd, bd in (("text", "results_deberta/text", "results/text"), ("deva_av", "results_deberta/deva_av", "results/deva_av"),
                   ("mag", "results_deberta_ausoft/mag", "results_ausoft/mag"), ("au_soft", "results_deberta_ausoft/soft", "results_ausoft/soft"),
                   ("au_soft_emo", "results_deberta_acc/soft_emo", "results_acc/soft_emo")):
    for tag, d in (("DeBERTa", dd), ("BERT", bd)):
        base, v = os.path.split(d)
        pr = {int(re.search(r"seed(\d+)_test", p).group(1)): pd.read_csv(p).set_index("key") for p in glob.glob(os.path.join(R, "mosi", base, f"{v}_seed*_test_pred.csv"))}
        if pr:
            ms = [mets(te.label.values, p.pred.loc[te.index].values, "mosi") for p in pr.values()]
            print(f"{nm:12s} {tag:8s} n={len(pr)} " + " ".join(f"{k} {np.mean([x[k] for x in ms]):.4f}" for k in ("acc2", "acc2_non0", "acc7", "mae", "corr")))
dt = {int(re.search(r"seed(\d+)_test", p).group(1)): pd.read_csv(p).set_index("key") for p in glob.glob(os.path.join(R, "mosi/results_deberta/text_seed*_test_pred.csv"))}
de = {int(re.search(r"seed(\d+)_test", p).group(1)): pd.read_csv(p).set_index("key") for p in glob.glob(os.path.join(R, "mosi/results_deberta_acc/soft_emo_seed*_test_pred.csv"))}
seeds = sorted(set(dt) & set(de))
for k in ("acc2", "acc2_non0", "mae", "corr"):
    a = [mets(te.label.values, de[s].pred.loc[te.index].values, "mosi")[k] for s in seeds]; b = [mets(te.label.values, dt[s].pred.loc[te.index].values, "mosi")[k] for s in seeds]
    print(f"  DeBERTa au_soft_emo vs DeBERTa text {k}: {np.mean(a) - np.mean(b):+.4f} (p={ttest_rel(a, b).pvalue:.3f})")
