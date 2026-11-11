"""CMU-MOSI results: MMSA metrics (Acc-2 has-0 / non-0, F1 both, Acc-7, MAE, Corr), mean±std over seeds,
per-seed paired t-test of au_soft(+emo) vs each baseline. Run with MSA_DATA=outputs/mosi MSA_DATASET=mosi."""
import glob
import json
import os

import numpy as np
from scipy.stats import ttest_rel

from train_bert import DATASET, FULL

ROWS = [("text (BERT)", "results", "cond", "text"), ("DEVA-style (en)", "results", "cond", "deva_av"),
        ("MAG-BERT-style", "results_ausoft", "variant", "mag"), ("au_soft", "results_ausoft", "variant", "soft"),
        ("au_soft + emo", "results_acc", "variant", "soft_emo")]
M = ["acc2", "f1", "acc2_non0", "f1_non0", "acc7", "mae", "corr"]


def load(d, key, name):
    return {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(FULL, d, "*_seed*.json"))
            if json.load(open(p))[key] == name}


res = {n: load(d, k, v) for n, d, k, v in ROWS}
lines = [f"### {DATASET.upper()} test, mean±std over seeds",
         "| model | seeds | Acc-2 (has-0) | F1 (has-0) | Acc-2 (non-0) | F1 (non-0) | Acc-7 | MAE | Corr |", "|---|---|---|---|---|---|---|---|---|"]
for n, js in res.items():
    if js:
        rs = list(js.values())
        lines.append(f"| {n} | {len(rs)} | " + " | ".join(
            f"{np.mean([r['test'][m] for r in rs]):.4f}±{np.std([r['test'][m] for r in rs], ddof=1) if len(rs) > 1 else 0:.4f}" for m in M) + " |")
lines += ["\n### per-seed paired t-test p (diff = ours − baseline)", "| ours | baseline | " + " | ".join(M) + " |", "|---" * (len(M) + 2) + "|"]
for ours in ("au_soft", "au_soft + emo"):
    for base in ("text (BERT)", "DEVA-style (en)", "MAG-BERT-style", "au_soft"):
        if base == ours or not res[ours] or not res[base]:
            continue
        seeds = sorted(set(res[ours]) & set(res[base]))
        if len(seeds) < 2:
            continue
        cells = []
        for m in M:
            a = [res[ours][s]["test"][m] for s in seeds]; b = [res[base][s]["test"][m] for s in seeds]
            cells.append(f"{np.mean(a) - np.mean(b):+.4f} (p={ttest_rel(a, b).pvalue:.3f})")
        lines.append(f"| {ours} | {base} | " + " | ".join(cells) + " |")
txt = "\n".join(lines)
open(os.path.join(FULL, f"{DATASET}_results.md"), "w").write(txt)
print(txt)
