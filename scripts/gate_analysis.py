"""Gate results table (mean±std over seeds) next to text / tmpl_au / DEVA-style rows, plus gate-value analysis."""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

F = os.path.expanduser("~/work/msa-pilot/outputs/full")
rows = []
for p in glob.glob(os.path.join(F, "results", "*_seed*.json")):
    r = json.load(open(p))
    if r["cond"] in ("text", "tmpl_au", "tmpl_vis", "deva_v", "deva_av"):
        rows.append((r["cond"], r))
for p in glob.glob(os.path.join(F, "results_gate", "*_seed*.json")):
    r = json.load(open(p)); rows.append(("gate:" + r["variant"], r))
order = ["text", "deva_v", "deva_av", "tmpl_au", "tmpl_vis", "gate:nogate", "gate:gate", "gate:gate_aux"]
lines = []
for sub in ("test", "test_conflict", "test_align"):
    lines += [f"\n### {sub}", "| model | seeds | Acc-2 | F1 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for name in order:
        rs = [r[sub] for n, r in rows if n == name]
        if not rs:
            continue
        f = lambda m: f"{np.mean([x[m] for x in rs]):.4f}±{np.std([x[m] for x in rs], ddof=1) if len(rs) > 1 else 0:.4f}"
        lines.append(f"| {name} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")

# ---- gate-value analysis on test predictions
ana = []
for p in sorted(glob.glob(os.path.join(F, "results_gate", "*_test_pred.csv"))):
    v = os.path.basename(p).split("_seed")[0]
    if v == "nogate":
        continue
    d = pd.read_csv(p)
    tv = (np.sign(d.label_V) != np.sign(d.label_T)).astype(int)
    gap = (d.label_V - d.label_T).abs()
    e_err = (d.y_e - d.label).abs(); t_err = (d.y_t - d.label).abs()
    ana.append({"variant": v, "seed": os.path.basename(p).split("_seed")[1].split("_")[0],
                "gate_mean": d.gate.mean(), "gate_std": d.gate.std(),
                "gate_conflict": d.gate[d.category == "conflict"].mean(),
                "gate_align": d.gate[d.category != "conflict"].mean(),
                "spearman_gate_|V-T|": spearmanr(d.gate, gap)[0],
                "auroc_gate_TVdisagree": roc_auc_score(tv, d.gate),
                "auroc_conflicthead_TVdisagree": roc_auc_score(tv, d.conflict_logit),
                "auroc_conflicthead_category": roc_auc_score((d.category == "conflict").astype(int), d.conflict_logit),
                # does the gate open when the evidence branch is the better predictor?
                "spearman_gate_(t_err-e_err)": spearmanr(d.gate, t_err - e_err)[0]})
A = pd.DataFrame(ana)
summ = A.drop(columns="seed").groupby("variant").agg(["mean", "std"]).round(3).T if len(A) else A
txt = "\n".join(lines) + "\n\n### gate analysis (test, mean/std over seeds)\n" + (summ.to_string() if len(A) else "n/a")
open(os.path.join(F, "gate_results.md"), "w").write(txt)
A.to_csv(os.path.join(F, "gate_analysis_per_seed.csv"), index=False)
print(txt)
