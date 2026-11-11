"""Frame densification: table + significance vs 8-frame au_soft+emo (and au_soft)."""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import FULL, metrics


def load(d, v):
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(FULL, d, f"{v}_seed*.json"))}
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(FULL, d, f"{v}_seed*_test_pred.csv")))]
    return js, (pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(1)) if pr else None)


def boot(a, b, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[metric] - metrics(y, pb)[metric]
    d = np.array([metrics(y[s], pa[s])[metric] - metrics(y[s], pb[s])[metric] for s in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


bases = {"au_soft+emo (8 full frames)": load("results_acc", "soft_emo"), "au_soft (8 full frames)": load("results_ausoft", "soft")}
V = ["crop8", "seg8", "dense16"]
lines = []
for sub in ("test", "test_conflict", "test_align"):
    lines += [f"\n### {sub}", "| model | seeds | Acc-2 | F1 | MAE | Corr | sec/run |", "|---|---|---|---|---|---|---|"]
    for name, (js, _) in list(bases.items()) + [(v, load("results_dense", v)) for v in V]:
        rs = list(js.values())
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            lines.append(f"| {name} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} | {np.mean([r.get('sec', np.nan) for r in rs]):.0f} |")
lines += ["\n### significance (diff = variant − base; per-seed paired t; 2k paired bootstrap on seed-averaged preds)",
          "| variant | base | subset | metric | diff | t p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|---|"]
for v in V:
    js, av = load("results_dense", v)
    if av is None or len(js) < 2:
        continue
    for bname, (bjs, bav) in bases.items():
        seeds = sorted(set(js) & set(bjs))
        for sub in ("full", "conflict", "align"):
            key = "test" if sub == "full" else f"test_{sub}"
            a = av if sub == "full" else av[(av.category == "conflict") == (sub == "conflict")]
            for m in ("acc2", "mae", "corr"):
                tp = ttest_rel([js[s][key][m] for s in seeds], [bjs[s][key][m] for s in seeds]).pvalue
                obs, lo, hi, bp = boot(a, bav, m)
                lines.append(f"| {v} | {bname} | {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
txt = "\n".join(lines)
open(os.path.join(FULL, "dense_results.md"), "w").write(txt)
print(txt)
