"""DIFF v2: table, significance vs au_soft and vs soft_diff (F), diagnostics."""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import metrics

F = os.path.expanduser("~/work/msa-pilot/outputs/full")


def load(d, v):
    js = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{v}_seed*.json"))}
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(F, d, f"{v}_seed*_test_pred.csv")))]
    avg = pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(1)) if pr else None
    return js, avg


def boot(a, b, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[metric] - metrics(y, pb)[metric]
    d = np.array([metrics(y[s], pa[s])[metric] - metrics(y[s], pb[s])[metric] for s in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, *np.percentile(d, [2.5, 97.5]), min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


bases = {"au_soft": load("results_ausoft", "soft"), "soft_diff": load("results_acc", "soft_diff")}
V = ["diff2_nosup", "diff2_uni", "diff2_full"]
lines = []
for sub in ("test", "test_conflict", "test_align"):
    lines += [f"\n### {sub}", "| model | seeds | Acc-2 | F1 | MAE | Corr |", "|---|---|---|---|---|---|"]
    for name, (js, _) in list(bases.items()) + [(v, load("results_diff2", v)) for v in V]:
        rs = list(js.values())
        if len(rs) > 1:
            f = lambda m: f"{np.mean([r[sub][m] for r in rs]):.4f}±{np.std([r[sub][m] for r in rs], ddof=1):.4f}"
            lines.append(f"| {name} | {len(rs)} | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} |")
lines += ["\n### significance (diff = variant − base; per-seed paired t; 2k paired bootstrap on seed-averaged preds)",
          "| variant | base | subset | metric | diff | t p | bootstrap 95% CI | bootstrap p |", "|---|---|---|---|---|---|---|---|"]
diag = ["\n### diagnostics (test; mean±std over seeds)",
        "| variant | shuffle ΔAcc-2 full | shuffle ΔMAE full | shuffle ΔCorr full | shuffle ΔAcc-2 Conflict | shuffle ΔCorr Conflict | mean |Δpred| | Spearman(‖z_v−z_t‖, |T−V|) | AUROC sign conflict |",
        "|---|---|---|---|---|---|---|---|---|"]
for v in V:
    js, av = load("results_diff2", v)
    if av is None or len(js) < 2:
        continue
    for bname, (bjs, bav) in bases.items():
        if bav is None:
            continue
        seeds = sorted(set(js) & set(bjs))
        for sub in ("full", "conflict", "align"):
            key = "test" if sub == "full" else f"test_{sub}"
            a = av if sub == "full" else av[(av.category == "conflict") == (sub == "conflict")]
            for m in ("acc2", "mae", "corr"):
                tp = ttest_rel([js[s][key][m] for s in seeds], [bjs[s][key][m] for s in seeds]).pvalue
                obs, lo, hi, bp = boot(a, bav, m)
                lines.append(f"| {v} | {bname} | {sub} | {m} | {obs:+.4f} | {tp:.4f} | [{lo:+.4f}, {hi:+.4f}] | {bp:.4f} |")
    ds = [r["diag"] for r in js.values()]
    f = lambda path: (lambda xs: f"{np.mean(xs):+.4f}±{np.std(xs, ddof=1):.4f}")([eval("d" + path) for d in ds])
    diag.append(f"| {v} | {f('[\"shuffle_delta_full\"][\"acc2\"]')} | {f('[\"shuffle_delta_full\"][\"mae\"]')} | {f('[\"shuffle_delta_full\"][\"corr\"]')} | "
                f"{f('[\"shuffle_delta_conflict\"][\"acc2\"]')} | {f('[\"shuffle_delta_conflict\"][\"corr\"]')} | {f('[\"mean_abs_dpred_shuffle\"]')} | "
                f"{f('[\"spearman_gap_vs_trueTVgap\"]')} | {f('[\"auroc_gap_sign_conflict\"]')} |")
txt = "\n".join(lines + diag)
open(os.path.join(F, "diff2_results.md"), "w").write(txt)
print(txt)
