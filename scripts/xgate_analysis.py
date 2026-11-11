"""Results table for the single-cross-encoder variants vs baselines + evidence-gate analysis."""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

F = os.path.expanduser("~/work/msa-pilot/outputs/full")
rows = []
for p in glob.glob(os.path.join(F, "results", "*_seed*.json")):
    r = json.load(open(p))
    if r["cond"] in ("text", "tmpl_au", "deva_av"):
        rows.append((r["cond"], r))
for p in glob.glob(os.path.join(F, "results_gate", "gate_aux_seed*.json")):
    rows.append(("dual:gate_aux", json.load(open(p))))
for p in glob.glob(os.path.join(F, "results_xgate", "*_seed*.json")):
    r = json.load(open(p)); rows.append((r["variant"], r))
order = ["text", "tmpl_au", "deva_av", "dual:gate_aux", "zh_concat", "xgate_noaux", "xgate", "zh_noid", "zh_code_noid", "zh_perm_noid"]
lines = []
for sub in ("test", "test_conflict", "test_align"):
    lines += [f"\n### {sub}", "| model | seeds | Acc-2 | F1 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for name in order:
        rs = [r[sub] for n, r in rows if n == name]
        if rs:
            f = lambda m: f"{np.mean([x[m] for x in rs]):.4f}±{np.std([x[m] for x in rs], ddof=1) if len(rs) > 1 else 0:.4f}"
            lines.append(f"| {name} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
ana = []
for p in sorted(glob.glob(os.path.join(F, "results_xgate", "*_test_pred.csv"))):
    v = os.path.basename(p).split("_seed")[0]
    if v == "zh_concat":
        continue
    d = pd.read_csv(p)
    tv = (np.sign(d.label_V) != np.sign(d.label_T)).astype(int)
    ana.append({"variant": v, "gate_mean": d.gate.mean(), "gate_std": d.gate.std(),
                "gate_conflict": d.gate[d.category == "conflict"].mean(),
                "gate_align": d.gate[d.category != "conflict"].mean(),
                "auroc_gate_TVdisagree": roc_auc_score(tv, d.gate),
                "auroc_gate_category": roc_auc_score((d.category == "conflict").astype(int), d.gate),
                "spearman_gate_|V-T|": spearmanr(d.gate, (d.label_V - d.label_T).abs())[0],
                "auroc_conflicthead_TVdisagree": roc_auc_score(tv, d.conflict_logit),
                "auroc_conflicthead_category": roc_auc_score((d.category == "conflict").astype(int), d.conflict_logit)})
A = pd.DataFrame(ana)
txt = "\n".join(lines) + "\n\n### evidence-gate analysis (test; mean/std over seeds)\n" + \
      (A.groupby("variant").agg(["mean", "std"]).round(3).T.to_string() if len(A) else "n/a")
open(os.path.join(F, "xgate_results.md"), "w").write(txt)
print(txt)
