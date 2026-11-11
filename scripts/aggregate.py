"""Aggregate results/*.json -> mean±std table (markdown + csv)."""
import glob
import json
import os

import pandas as pd

FULL = os.path.expanduser("~/work/msa-pilot/outputs/full")
ORDER = ["text", "tmpl_au", "tmpl_head", "tmpl_gaze", "tmpl_vis", "tmpl_vis_shuf", "tmpl_pros", "vlm8b", "vlmv8b", "shuf8b", "vlm4b_zs", "vlm4b_sft", "tmpl", "tmpl_v3b", "vlm", "vlmv", "tmpl_vlmv", "shuf"]
rows = []
for p in glob.glob(os.path.join(FULL, "results", "*_seed*.json")):
    r = json.load(open(p))
    for sub in ("test", "test_conflict", "test_align"):
        rows.append({"cond": r["cond"], "seed": r["seed"], "subset": sub.replace("test_", "") if sub != "test" else "full",
                     **{m: r[sub][m] for m in ("acc2", "f1", "mae", "corr")}, "n": r[sub]["n"]})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(FULL, "results_all_seeds.csv"), index=False)
lines = []
for sub in ("full", "conflict", "align"):
    d = df[df.subset == sub]
    n = int(d.n.iloc[0]) if len(d) else 0
    lines += [f"\n### test {sub} (n={n})\n", "| cond | seeds | Acc-2 | F1 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for c in [c for c in ORDER if c in set(d.cond)]:
        g = d[d.cond == c]
        f = lambda m: f"{g[m].mean():.4f}±{g[m].std(ddof=1) if len(g) > 1 else 0:.4f}"
        lines.append(f"| {c} | {len(g)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
txt = "\n".join(lines)
open(os.path.join(FULL, "results_table.md"), "w").write(txt)
print(txt)
