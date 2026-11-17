"""Adjudicator pilot (llm_diff.py au_res / au_adj, --exp m) against the late-average baseline, weights chosen on VALID only.
Baseline: (1-w) * [LLM + face tokens] + w * mean(text, face, speech expert reading); w in 0..0.6 picked per seed by valid MAE.
The LLM + face runs come from outputs/full/llm_val (same recipe as llm_pilot au, re-run to store valid predictions).
Subsets by expert disagreement (no unimodal label): sign split among the three readings, and the top third by std of the readings.
Writes outputs/full/llm_pilot/adj_results.md"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import FULL, metrics

D, V = os.path.join(FULL, "llm_pilot"), os.path.join(FULL, "llm_val")
rows = {}
for s in range(1, 11):
    f = {n: os.path.join(D, f"{n}_m_seed{s}_test_pred.csv") for n in ("au_res", "au_adj")}
    fa = os.path.join(V, f"au_seed{s}_test_pred.csv")
    if not (os.path.exists(fa) and all(os.path.exists(x) for x in f.values())):
        continue
    adj, res, au = pd.read_csv(f["au_adj"]), pd.read_csv(f["au_res"]), pd.read_csv(fa)
    adjv, auv = pd.read_csv(f["au_adj"].replace("_test_", "_valid_")), pd.read_csv(fa.replace("_test_", "_valid_"))
    assert (adj.key == au.key).all() and (adjv.key == auv.key).all() and (res.key == au.key).all()
    E, Ev = adj[["rt", "rv", "ra"]].values, adjv[["rt", "rv", "ra"]].values
    ws = np.arange(0, 0.61, 0.05)
    w = ws[int(np.argmin([np.abs((1 - x) * auv.pred_full.values + x * Ev.mean(1) - auv.label.values).mean() for x in ws]))]
    P = {"LLM + face tokens": au.pred_full.values, "mean of 3 experts": E.mean(1),
         "late average (w on valid)": (1 - w) * au.pred_full.values + w * E.mean(1),
         "residual on expert mean (au_res)": res.pred_full.values, "adjudicator: residual + closest-expert head (au_adj)": adj.pred_full.values}
    sg = E > 0; split = ~(sg.all(1) | (~sg).all(1)); d = E.std(1)
    subs = {"all": np.ones(len(au), bool), "experts split sign": split, "experts same sign": ~split, "top-third disagreement": d >= np.quantile(d, 2 / 3),
            "conflict (unimodal labels)": (au.category == "conflict").values}
    for n, p in P.items():
        for sn, m in subs.items():
            rows.setdefault((sn, n), []).append(metrics(au.label.values[m], p[m]))
    rows.setdefault(("w", "w"), []).append(w)
L = [f"seeds: {len(rows.get(('w', 'w'), []))}; late-average w per seed (valid): {rows.get(('w', 'w'), [])}", ""]
for sn in ("all", "experts split sign", "experts same sign", "top-third disagreement", "conflict (unimodal labels)"):
    L += [f"### {sn}", "| model | Acc-2 | MAE | Corr |", "|---|---|---|---|"]
    names = [n for (s_, n) in rows if s_ == sn]
    for n in names:
        v = rows[(sn, n)]; L.append(f"| {n} | {np.mean([x['acc2'] for x in v]) * 100:.2f} | {np.mean([x['mae'] for x in v]):.4f} | {np.mean([x['corr'] for x in v]):.4f} |")
    if names and len(rows[(sn, names[0])]) > 1:
        for A, B in ((names[-1], "late average (w on valid)"), (names[-1], names[-2]), (names[-1], "LLM + face tokens")):
            L.append(f"- [{A}] - [{B}]: " + "; ".join(
                f"{k} {np.mean([x[k] - y[k] for x, y in zip(rows[(sn, A)], rows[(sn, B)])]):+.4f} (p={ttest_rel([x[k] for x in rows[(sn, A)]], [x[k] for x in rows[(sn, B)]]).pvalue:.3f})"
                for k in ("acc2", "mae", "corr")))
    L.append("")
open(os.path.join(D, "adj_results.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
