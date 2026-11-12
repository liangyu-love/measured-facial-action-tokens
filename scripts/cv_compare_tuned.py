"""5-fold CV comparison on CH-SIMS v2 under the TUNED configurations (run_cv3.sh). Same statistics as cv_compare.py: pooled out-of-fold
predictions per seed (all 4403 clips), per-seed mean Acc-2, seed-ensemble Acc-2, paired clip-bootstrap 95% CI / p of the ensemble
difference of au_soft+emo over each method, paired t-test over seeds. Adds Corr / MAE and the strict subset without overall label 0.
-> outputs/full/cv_compare_tuned.json"""
import json, os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

R = os.path.expanduser("~/work/msa-pilot/outputs"); CV = os.path.join(R, "full_cv")
c = pd.read_csv(os.path.join(R, "full", "clips.csv"), dtype={"video_id": str, "clip_id": str}).set_index("key")
strict = (np.sign(c.label_T) * np.sign(c.label_V)) < 0
SUB = {"all": c.index, "conflict": c.index[c.category == "conflict"], "strict": c.index[strict],
       "strict_non0": c.index[strict & (c.label != 0)]}
M = {"text": "cvt/text/text_seed{s}_test_pred.csv", "deva_dense": "cvt/deva_dense/deva_av_seed{s}_test_pred.csv",
     "mag": "cvt/mag/mag_seed{s}_test_pred.csv", "au_soft_emo": "cvt/au_soft_emo/soft_emo_seed{s}_test_pred.csv"}


def pooled(m, s):
    parts = []
    for k in range(5):
        f = os.path.join(CV, f"fold{k}", M[m].format(s=s))
        if not os.path.exists(f):
            return None
        parts.append(pd.read_csv(f).set_index("key"))
    d = pd.concat(parts)
    assert d.index.is_unique and len(d) == len(c), (m, s, len(d))
    return d.loc[c.index]


P = {m: {s: pooled(m, s) for s in range(1, 6)} for m in M}
P = {m: {s: d for s, d in v.items() if d is not None} for m, v in P.items()}
y = c.label.values
ix = lambda idx: c.index.get_indexer(idx)
acc = lambda p, idx: float((((y > 0) == (np.clip(p, -1, 1) > 0))[ix(idx)]).mean())
corr = lambda p, idx: float(np.corrcoef(np.clip(p, -1, 1)[ix(idx)], y[ix(idx)])[0, 1])
mae = lambda p, idx: float(np.abs(np.clip(p, -1, 1)[ix(idx)] - y[ix(idx)]).mean())
res = {"n": {k: int(len(v)) for k, v in SUB.items()}}
rng = np.random.default_rng(0)
ens = {m: np.mean([d.pred.values for d in v.values()], axis=0) for m, v in P.items() if v}
for m, v in P.items():
    if not v:
        continue
    for sname, idx in SUB.items():
        per = [acc(d.pred.values, idx) for d in v.values()]
        res[f"{m}|{sname}"] = {"seeds": len(per), "per_seed_mean": float(np.mean(per)), "per_seed_std": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                               "ensemble": acc(ens[m], idx), "corr": float(np.mean([corr(d.pred.values, idx) for d in v.values()])),
                               "mae": float(np.mean([mae(d.pred.values, idx) for d in v.values()]))}
ref = "au_soft_emo"
for m in ("text", "deva_dense", "mag"):
    if m not in ens or ref not in ens:
        continue
    for sname, idx in SUB.items():
        ii = ix(idx); yy = y[ii] > 0
        ca = (yy == (np.clip(ens[ref][ii], -1, 1) > 0)).astype(float); cb = (yy == (np.clip(ens[m][ii], -1, 1) > 0)).astype(float)
        diff = ca - cb; bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
        seeds = sorted(set(P[ref]) & set(P[m]))
        pa = [acc(P[ref][s].pred.values, idx) for s in seeds]; pb = [acc(P[m][s].pred.values, idx) for s in seeds]
        ka = [corr(P[ref][s].pred.values, idx) for s in seeds]; kb = [corr(P[m][s].pred.values, idx) for s in seeds]
        res[f"{ref}-{m}|{sname}"] = {"ens_diff": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                    "boot_p": float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), "fixed_broken": [int((diff > 0).sum()), int((diff < 0).sum())],
                                    "per_seed_diff": float(np.mean(pa) - np.mean(pb)), "t_p": float(ttest_rel(pa, pb).pvalue) if len(seeds) > 1 else None,
                                    "corr_diff": float(np.mean(ka) - np.mean(kb)), "corr_t_p": float(ttest_rel(ka, kb).pvalue) if len(seeds) > 1 else None}
for k, v in res.items():
    print(k, json.dumps(v))
json.dump(res, open(os.path.join(R, "full", "cv_compare_tuned.json"), "w"), indent=1)
