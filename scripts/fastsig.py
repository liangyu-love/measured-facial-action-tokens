"""Fast significance for the key DIFF3 comparisons (vectorised bootstrap, acc2/mae/corr; SIMS v2 conventions)."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
F = os.path.expanduser("~/work/msa-pilot/outputs/full")
def load(d, v):
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(f"{F}/{d}/{v}_seed*.json")}
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(f"{F}/{d}/{v}_seed*_test_pred.csv"))]
    return js, pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(axis=1))
def met(y, p):  # rows = bootstrap replicates
    p = np.clip(p, -1, 1)
    acc = ((y > 0) == (p > 0)).mean(-1); mae = np.abs(y - p).mean(-1)
    yc, pc = y - y.mean(-1, keepdims=True), p - p.mean(-1, keepdims=True)
    corr = (yc * pc).sum(-1) / np.sqrt((yc ** 2).sum(-1) * (pc ** 2).sum(-1))
    return {"acc2": acc, "mae": mae, "corr": corr}
M = {"au_soft+emo": ("results_acc", "soft_emo"), "diff2_pre": ("results_pre", "diff2_pre")}
for v in ("diff3_unc", "diff3_conf", "diff3_texp", "diff3_frame", "diff3_best"):
    ja, a = load("results_diff3", v)
    for ref, (d, rv) in M.items():
        jb, b = load(d, rv)
        seeds = sorted(set(ja) & set(jb))
        for s in ("full", "conflict", "align"):
            m = np.ones(len(a), bool) if s == "full" else ((a.category == "conflict") if s == "conflict" else a.category.isin(["equal", "spne"])).values
            y, pa, pb = a.label.values[m], a.pred.values[m], b.loc[a.index].pred.values[m]
            idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
            ma, mb = met(y[idx], pa[idx]), met(y[idx], pb[idx]); o_a, o_b = met(y[None], pa[None]), met(y[None], pb[None])
            key = "test" if s == "full" else f"test_{s}"; cells = []
            for k in ("acc2", "mae", "corr"):
                dd = ma[k] - mb[k]; bp = min(1, max(2 * min((dd < 0).mean(), (dd > 0).mean()), 1 / 2000))
                tp = ttest_rel([ja[x][key][k] for x in seeds], [jb[x][key][k] for x in seeds]).pvalue
                cells.append(f"{k} {float((o_a[k] - o_b[k])[0]):+.4f} (t {tp:.3f}, boot {bp:.3f})")
            print(f"{v} vs {ref} [{s}] n={len(seeds)}: " + "; ".join(cells))
