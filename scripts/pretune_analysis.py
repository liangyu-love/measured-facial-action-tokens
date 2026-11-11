"""(1) RoBERTa vs BERT backbone, (2) diff3_tri vs diff3_unc. 5-seed means; per-seed t + vectorised bootstrap on seed-averaged
predictions; prints 'BACKBONE roberta|bert' (roberta only if it wins valid MAE for ALL four methods). -> outputs/full/pretune_results.md"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

F = os.path.expanduser("~/work/msa-pilot/outputs/full")


def load(d, v):
    js = {}
    for p in glob.glob(f"{F}/{d}/{v}_seed*.json"):
        r = json.load(open(p))
        if "test" in r and r.get("seed", 0) <= 5:
            js[r["seed"]] = r
    pr = [pd.read_csv(f"{F}/{d}/{v}_seed{s}_test_pred.csv").set_index("key") for s in sorted(js)]
    av = pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(axis=1)) if pr else None
    return js, av


def met(y, p):
    p = np.clip(p, -1, 1); acc = ((y > 0) == (p > 0)).mean(-1); mae = np.abs(y - p).mean(-1)
    yc, pc = y - y.mean(-1, keepdims=True), p - p.mean(-1, keepdims=True)
    return {"acc2": acc, "mae": mae, "corr": (yc * pc).sum(-1) / np.sqrt((yc ** 2).sum(-1) * (pc ** 2).sum(-1))}


def sig(A, B):
    (ja, a), (jb, b) = A, B; out = []; seeds = sorted(set(ja) & set(jb))
    for s in ("full", "conflict", "align"):
        m = np.ones(len(a), bool) if s == "full" else ((a.category == "conflict") if s == "conflict" else a.category.isin(["equal", "spne"])).values
        y, pa, pb = a.label.values[m], a.pred.values[m], b.loc[a.index].pred.values[m]
        idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
        ma, mb = met(y[idx], pa[idx]), met(y[idx], pb[idx]); oa, ob = met(y[None], pa[None]), met(y[None], pb[None])
        key = "test" if s == "full" else f"test_{s}"; cells = []
        for k in ("acc2", "mae", "corr"):
            dd = ma[k] - mb[k]; bp = min(1, max(2 * min((dd < 0).mean(), (dd > 0).mean()), 1 / 2000))
            tp = ttest_rel([ja[x][key][k] for x in seeds], [jb[x][key][k] for x in seeds]).pvalue if len(seeds) > 1 else float("nan")
            cells.append(f"{float((oa[k] - ob[k])[0]):+.4f} ({tp:.3f}/{bp:.3f})")
        out.append(f"{s}: " + "; ".join(f"{k} {c}" for k, c in zip(("Acc-2", "MAE", "Corr"), cells)))
    return out


def row(n, js):
    keys = (("valid", "mae"), ("test", "acc2"), ("test", "mae"), ("test", "corr"), ("test_conflict", "acc2"), ("test_conflict", "corr"), ("test_align", "acc2"))
    return f"| {n} | {len(js)} | " + " | ".join(f"{np.mean([r[k][m] for r in js.values()]):.4f}" for k, m in keys) + " |"


H = ["| model | seeds | valid MAE | Acc-2 | MAE | Corr | Conflict Acc-2 | Conflict Corr | Align Acc-2 |", "|---|---|---|---|---|---|---|---|---|"]
pairs = {"text": (("results", "text"), ("results_rob2", "text")), "au_soft+emo": (("results_acc", "soft_emo"), ("results_rob2", "soft_emo")),
         "diff2_pre": (("results_bert2", "diff2_pre"), ("results_rob2", "diff2_pre")), "diff3_unc": (("results_bert2", "diff3_unc"), ("results_rob2", "diff3_unc"))}
L = ["## (1) Backbone: bert-base-chinese vs chinese-roberta-wwm-ext (CH-SIMS v2, 5 seeds; same tuned experts)"] + H
wins, data = [], {}
for n, (b, r) in pairs.items():
    B, R = load(*b), load(*r); data[n] = (B, R)
    if B[0]:
        L.append(row(f"{n} (BERT)", B[0]))
    if R[0]:
        L.append(row(f"{n} (RoBERTa)", R[0]))
    if B[0] and R[0]:
        wins.append(bool(np.mean([x["valid"]["mae"] for x in R[0].values()]) < np.mean([x["valid"]["mae"] for x in B[0].values()])))
L.append("\nRoBERTa − BERT, diff (t p / bootstrap p):")
for n, (B, R) in data.items():
    if B[0] and R[0]:
        L += [f"- {n} {x}" for x in sig(R, B)]
bb = "roberta" if len(wins) == 4 and all(wins) else "bert"
L.append(f"\nvalid-MAE wins for RoBERTa (text, au_soft+emo, diff2_pre, diff3_unc): {wins} -> backbone: **{bb}**")
U, T, E = load("results_bert2", "diff3_unc"), load("results_bert2", "diff3_tri"), load("results_acc", "soft_emo")
L += ["\n## (2) Audio expert: diff3_tri (text + visual + audio experts) vs diff3_unc (BERT, tuned experts)"] + H
for n, x in (("au_soft+emo", E), ("diff3_unc", U), ("diff3_tri", T)):
    if x[0]:
        L.append(row(n, x[0]))
if T[0] and U[0]:
    L += ["\ndiff3_tri − diff3_unc:"] + [f"- {x}" for x in sig(T, U)] + ["\ndiff3_tri − au_soft+emo:"] + [f"- {x}" for x in sig(T, E)]
    dg = [r["diag"] for r in T[0].values()]
    g = lambda k, s=None: np.mean([d[k][s] if s else d[k] for d in dg if k in d])
    L.append(f"\ndiagnostics diff3_tri: shuffle-DIFF dCorr full {g('shuffle_diff_delta_full', 'corr'):+.3f} / Conflict {g('shuffle_diff_delta_conflict', 'corr'):+.3f}; "
             f"AUROC T/V sign conflict {g('auroc_gap_sign_conflict'):.3f}; AUROC T/A sign conflict {g('auroc_gapTA_sign_conflictTA'):.3f}; "
             f"Spearman |ya-yt| vs |T-A| {g('spearman_gapTA_vs_trueTAgap'):+.3f}; audio expert ya~label_A {g('ya_corr_labelA'):.3f}")
txt = "\n".join(L)
open(f"{F}/pretune_results.md", "w").write(txt)
print(txt)
print("BACKBONE", bb)
