"""Summarise llm_pilot.py and compare with the BERT runs of the same recipe family (miss_pilot std_seed*: full = au_soft+emo, zero = face
zeroed ~ text only). Writes outputs/full/llm_pilot/llm_results.md"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

from train_bert import FULL, metrics

D = os.path.join(FULL, "llm_pilot")
import re
L = lambda pat: [pd.read_csv(f) for f in sorted(glob.glob(pat), key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))]   # by seed number, so [:n] pairs seeds 1..n
src = {"LLM text": (L(os.path.join(D, "text_seed*_test_pred.csv")), "full"),
       "LLM + face tokens": (L(os.path.join(D, "au_seed*_test_pred.csv")), "full"),
       "LLM + face tokens, face zeroed": (L(os.path.join(D, "au_seed*_test_pred.csv")), "noV"),
       "LLM + face + DIFF(T,V)": (L(os.path.join(D, "au_diff_seed*_test_pred.csv")), "full"),
       "LLM + face + DIFF(T,V,A)": (L(os.path.join(D, "au_tri_seed*_test_pred.csv")), "full"),
       "LLM + face + DIFF(T,V,A) + trust head": (L(os.path.join(D, "au_tri_trust_seed*_test_pred.csv")), "full"),
       "  same, DIFF shuffled at test": (L(os.path.join(D, "au_tri_trust_seed*_test_pred.csv")), "shufD"),
       "  control: visual expert reading zeroed": (L(os.path.join(D, "au_tri_trust_v0_seed*_test_pred.csv")), "full"),
       "LLM + face + speech token (no experts)": (L(os.path.join(D, "au_spk_seed*_test_pred.csv")), "full"),
       "no unimodal labels: DIFF(T,V,A) + trust head": (L(os.path.join(D, "au_tri_trust_m_seed*_test_pred.csv")), "full"),
       "  no unimodal labels, control: visual reading zeroed": (L(os.path.join(D, "au_tri_trust_v0_m_seed*_test_pred.csv")), "full"),
       "LLM sees 8 frames":(L(os.path.join(D, "vid_seed*_test_pred.csv")), "full"),
       "LLM sees 8 frames + face tokens": (L(os.path.join(D, "vid_au_seed*_test_pred.csv")), "full"),
       "LLM sees 8 frames + face tokens, face zeroed": (L(os.path.join(D, "vid_au_seed*_test_pred.csv")), "noV"),
       "LLM sees 8 frames + expert token + trust head": (L(os.path.join(D, "vid_exp_seed*_test_pred.csv")), "full"),
       "  same, expert readings shuffled at test": (L(os.path.join(D, "vid_exp_seed*_test_pred.csv")), "shufD"),
       "LLM text, 10 ep": (L(os.path.join(FULL, "llm_ep10", "text_seed*_test_pred.csv")), "full"),
       "LLM + face tokens, 10 ep": (L(os.path.join(FULL, "llm_ep10", "au_seed*_test_pred.csv")), "full"),
       "BERT + face tokens (au_soft+emo)": (L(os.path.join(FULL, "miss_pilot", "std_seed*_test_pred.csv")), "full"),
       "BERT text (tuned)": (L(os.path.join(FULL, "tuned", "text", "text_seed[1-5]_test_pred.csv")), None)}


def val(name, sub, met):
    runs, c = src[name]; out = []
    for r in runs:
        m = np.ones(len(r), bool) if sub == "all" else ((r.category == "conflict").values if sub == "conflict" else (r.category != "conflict").values)
        out.append(metrics(r.label[m], r["pred_" + c if c else "pred"].values[m])[met])
    return np.array(out)


lines = ["| model | seeds | Acc-2 | MAE | Corr | conflict Acc-2 | conflict Corr | align Acc-2 |", "|---|---|---|---|---|---|---|---|"]
for n in src:
    if not src[n][0]:
        continue
    lines.append(f"| {n} | {len(src[n][0])} | {val(n, 'all', 'acc2').mean() * 100:.2f} | {val(n, 'all', 'mae').mean():.4f} | {val(n, 'all', 'corr').mean():.4f} | "
                 f"{val(n, 'conflict', 'acc2').mean() * 100:.2f} | {val(n, 'conflict', 'corr').mean():.4f} | {val(n, 'align', 'acc2').mean() * 100:.2f} |")
lines.append("")
if len(src["LLM text"][0]) == len(src["LLM + face tokens"][0]) > 1:
    for sub in ("all", "conflict"):
        for met in ("acc2", "mae", "corr"):
            x, y = val("LLM + face tokens", sub, met), val("LLM text", sub, met)
            b = val("BERT + face tokens (au_soft+emo)", sub, met)
            lines.append(f"- {sub} {met}: LLM+face - LLM text = {(x - y).mean():+.4f} (paired t p={ttest_rel(x, y).pvalue:.3f}); "
                         f"LLM+face - BERT+face = {x.mean() - b.mean():+.4f} (Welch p={ttest_ind(x, b, equal_var=False).pvalue:.3f})")
for A, B in (("LLM + face + DIFF(T,V)", "LLM + face tokens"), ("LLM + face + DIFF(T,V,A)", "LLM + face tokens"),
             ("LLM + face + DIFF(T,V,A) + trust head", "LLM + face tokens"),
             ("LLM + face + DIFF(T,V,A) + trust head", "  control: visual expert reading zeroed"),
             ("LLM + face + speech token (no experts)", "LLM + face tokens"),
             ("LLM + face + DIFF(T,V,A) + trust head", "LLM + face + speech token (no experts)"),
             ("no unimodal labels: DIFF(T,V,A) + trust head", "LLM + face tokens"),
             ("no unimodal labels: DIFF(T,V,A) + trust head", "LLM + face + speech token (no experts)"),
             ("no unimodal labels: DIFF(T,V,A) + trust head", "  no unimodal labels, control: visual reading zeroed"),
             ("LLM + face tokens", "LLM sees 8 frames"), ("LLM sees 8 frames + face tokens", "LLM sees 8 frames"),
             ("LLM sees 8 frames + expert token + trust head", "LLM sees 8 frames"),
             ("LLM + face tokens, 10 ep", "LLM + face tokens")):
    n = min(len(src[A][0]), len(src[B][0]))
    if n > 1:                                               # paired over the seeds both have (files are sorted by seed)
        for sub in ("all", "conflict"):
            lines.append(f"- {sub}: [{A}] - [{B}] ({n} seeds): " + "; ".join(
                f"{met} {(val(A, sub, met)[:n] - val(B, sub, met)[:n]).mean():+.4f} (p={ttest_rel(val(A, sub, met)[:n], val(B, sub, met)[:n]).pvalue:.3f})" for met in ("acc2", "mae", "corr")))
open(os.path.join(D, "llm_results.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
