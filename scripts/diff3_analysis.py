"""DIFF optimisation round summary. `--decide-base` -> prints '--texp' if V1 beats diff2_pre on mean valid MAE, else ''.
`--decide-best` -> writes results_diff3/best_flags.json (union of variants that beat diff2_pre on mean valid MAE) and prints it.
Default: writes outputs/full/diff3_results.md (tables, significance vs au_soft+emo and vs diff2_pre, diagnostics)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from train_bert import FULL, metrics

D3 = os.path.join(FULL, "results_diff3")
VARS = [("diff3_texp", "V1 text expert"), ("diff3_unc", "V2 uncertainty scalars"), ("diff3_frame", "V3 frame-level DIFF"),
        ("diff3_head", "V4 DIFF -> head"), ("diff3_rw05", "V5 reweight a=0.5"), ("diff3_rw1", "V5 reweight a=1"),
        ("diff3_ft", "V6 unfreeze visual"), ("diff3_moe", "V7 anchored mixture"), ("diff3_conf", "V8 conflict head"),
        ("diff3_best", "combined (valid-improving)")]
FLAG_OF = {"diff3_texp": {"texp": True}, "diff3_unc": {"unc": True}, "diff3_frame": {"frame": True}, "diff3_head": {"headcat": True},
           "diff3_rw05": {"rw": 0.5}, "diff3_rw1": {"rw": 1.0}, "diff3_ft": {"ft": True}, "diff3_moe": {"moe": True}, "diff3_conf": {"conf": True}}


def load(d, v):
    js = {}
    for p in glob.glob(os.path.join(d, f"{v}_seed*.json")):
        r = json.load(open(p))
        if "test" in r:
            js[r["seed"]] = r
    pr = [pd.read_csv(p).set_index("key") for p in sorted(glob.glob(os.path.join(d, f"{v}_seed*_test_pred.csv")))]
    av = pr[0][["label", "category"]].assign(pred=pd.concat([p.pred for p in pr], axis=1).mean(axis=1)) if pr else None
    return js, av


def vmae(js):
    return np.mean([r["valid"]["mae"] for r in js.values()]) if js else 9.0


BASE_JS, BASE_AV = load(os.path.join(FULL, "results_pre"), "diff2_pre")
EMO_JS, EMO_AV = load(os.path.join(FULL, "results_acc"), "soft_emo")

if "--decide-base" in sys.argv:
    print("--texp" if vmae(load(D3, "diff3_texp")[0]) < vmae(BASE_JS) else ""); sys.exit(0)
if "--decide-best" in sys.argv:
    b = vmae(BASE_JS); flags = {}
    good = {v: vmae(load(D3, v)[0]) for v in FLAG_OF if vmae(load(D3, v)[0]) < b}
    rws = {v: m for v, m in good.items() if v.startswith("diff3_rw")}
    for v in good:
        if v.startswith("diff3_rw") and v != min(rws, key=rws.get):
            continue
        fl = dict(FLAG_OF[v])
        if v in ("diff3_unc", "diff3_conf"):  # built on V1 when V1 helped
            fl.update(load(D3, v)[0][next(iter(load(D3, v)[0]))]["flags"])
            fl = {k: x for k, x in fl.items() if x}
        flags.update(fl)
    json.dump(flags, open(os.path.join(D3, "best_flags.json"), "w"))
    print(json.dumps({"base_valid_mae": round(b, 4), "improving": {k: round(x, 4) for k, x in good.items()}, "flags": flags})); sys.exit(0)


def boot(a, b, m, n=2000):
    rng = np.random.default_rng(0); y = a.label.values; pa, pb = a.pred.values, b.loc[a.index].pred.values
    obs = metrics(y, pa)[m] - metrics(y, pb)[m]
    d = np.array([metrics(y[i], pa[i])[m] - metrics(y[i], pb[i])[m] for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    return obs, min(1.0, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / n))


def sub(df, s):
    return df if s == "full" else (df[df.category == "conflict"] if s == "conflict" else df[df.category.isin(["equal", "spne"])])


rows = [("au_soft+emo", EMO_JS, EMO_AV), ("diff2_pre (base)", BASE_JS, BASE_AV)] + \
       [(f"{v} — {desc}", *load(D3, v)) for v, desc in VARS]
L = ["## DIFF optimisation round (CH-SIMS v2, 5 seeds, mean±std; selection by valid MAE)",
     "| model | valid MAE | Acc-2 | MAE | Corr | Conflict Acc-2 | Conflict MAE | Conflict Corr | Align Acc-2 | Align MAE |", "|---" * 10 + "|"]
fmt = lambda rs, k, m: f"{np.mean([r[k][m] for r in rs]):.4f}±{np.std([r[k][m] for r in rs], ddof=1) if len(rs) > 1 else 0:.4f}"
for name, js, _ in rows:
    if js:
        rs = list(js.values())
        L.append(f"| {name} | {vmae(js):.4f} | {fmt(rs, 'test', 'acc2')} | {fmt(rs, 'test', 'mae')} | {fmt(rs, 'test', 'corr')} | "
                 f"{fmt(rs, 'test_conflict', 'acc2')} | {fmt(rs, 'test_conflict', 'mae')} | {fmt(rs, 'test_conflict', 'corr')} | "
                 f"{fmt(rs, 'test_align', 'acc2')} | {fmt(rs, 'test_align', 'mae')} |")
if os.path.exists(os.path.join(D3, "best_flags.json")):
    L.append(f"\ndiff3_best flags: `{open(os.path.join(D3, 'best_flags.json')).read()}`")
L += ["\n### significance (diff, per-seed paired t p / bootstrap p on seed-averaged predictions)",
      "| model | vs | subset | ΔAcc-2 (t p / boot p) | ΔMAE | ΔCorr |", "|---|---|---|---|---|---|"]
for name, js, av in rows[2:]:
    if not js or av is None:
        continue
    for ref, rjs, rav in (("au_soft+emo", EMO_JS, EMO_AV), ("diff2_pre", BASE_JS, BASE_AV)):
        seeds = sorted(set(js) & set(rjs))
        for s in ("full", "conflict", "align"):
            key = "test" if s == "full" else f"test_{s}"; cells = []
            for m in ("acc2", "mae", "corr"):
                tp = ttest_rel([js[k][key][m] for k in seeds], [rjs[k][key][m] for k in seeds]).pvalue if len(seeds) > 1 else float("nan")
                o, bp = boot(sub(av, s), rav, m)
                cells.append(f"{o:+.4f} ({tp:.3f} / {bp:.3f})")
            L.append(f"| {name.split(' — ')[0]} | {ref} | {s} | " + " | ".join(cells) + " |")
L += ["\n### diagnostics (mean over seeds)", "| model | shuffle-DIFF ΔCorr full | ΔCorr Conflict | Spearman(gap,|T−V|) | AUROC sign conflict | yv~label_V | yt~label_T | moe w Conflict / Align |",
      "|---|---|---|---|---|---|---|---|"]
for name, js, _ in rows[1:]:
    if not js:
        continue
    dg = [r["diag"] for r in js.values()]
    g = lambda k, sk=None: np.mean([x[k][sk] if sk else x[k] for x in dg if k in x]) if any(k in x for x in dg) else float("nan")
    w = f"{g('moe_w_conflict'):.3f} / {g('moe_w_align'):.3f}" if any("moe_w_conflict" in x for x in dg) else "—"
    L.append(f"| {name.split(' — ')[0]} | {g('shuffle_diff_delta_full', 'corr'):+.4f} | {g('shuffle_diff_delta_conflict', 'corr'):+.4f} | "
             f"{g('spearman_gap_vs_trueTVgap'):+.3f} | {g('auroc_gap_sign_conflict'):.3f} | {g('yv_corr_labelV', None) if any('yv_corr_labelV' in x for x in dg) else float('nan'):.3f} | "
             f"{g('yt_corr_labelT') if any('yt_corr_labelT' in x for x in dg) else float('nan'):.3f} | {w} |")
txt = "\n".join(L)
open(os.path.join(FULL, "diff3_results.md"), "w").write(txt)
print(txt)
