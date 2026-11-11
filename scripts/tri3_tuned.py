"""Tuned 10-seed comparison for the three-expert variant (SIMS v2): per-seed paired t + paired bootstrap on seed-averaged predictions."""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
R = os.path.expanduser("~/work/msa-pilot/outputs/full"); T = os.path.join(R, "tuned")
c = pd.read_csv(os.path.join(R, "clips.csv")).set_index("key"); te = c[c.split == "test"]; sg = np.sign
S = {"all(1034)": te.index, "conflict(409)": te.index[te.category == "conflict"], "strict(93)": te.index[(sg(te.label_T) * sg(te.label_V)) < 0]}
def load(m):
    return {int(re.search(r"seed(\d+)_test", p).group(1)): pd.read_csv(p).set_index("key").pred for p in glob.glob(os.path.join(T, m, "*_seed*_test_pred.csv"))}
def met(y, p):
    p = np.clip(p, -1, 1); return {"acc2": ((y > 0) == (p > 0)).mean(), "mae": np.abs(y - p).mean(), "corr": np.corrcoef(y, p)[0, 1]}
P = {m: load(m) for m in ("diff3_tri3_trust", "diff3_oof", "diff3_unc", "au_soft_emo", "text")}
best = __import__("json").load(open(os.path.join(R, "tuned_configs.json")))
for m in ("diff3_tri3_trust", "diff3_oof"):
    print(m, "selected", best[m]["config"], "valid MAE", round(best[m]["valid_mae"], 4))
for m, pr in P.items():
    print(f"{m:17s} n={len(pr)} | " + " | ".join(f"{sn} " + " ".join(f"{k} {np.mean([met(te.label.loc[ix].values, p.loc[ix].values)[k] for p in pr.values()]):.4f}" for k in ("acc2", "mae", "corr")) for sn, ix in S.items()))
rng = np.random.default_rng(0)
for a, b in (("diff3_tri3_trust", "au_soft_emo"), ("diff3_tri3_trust", "diff3_unc"), ("diff3_tri3_trust", "diff3_oof")):
    seeds = sorted(set(P[a]) & set(P[b])); out = []
    for sn, ix in S.items():
        y = te.label.loc[ix].values; ea = pd.concat([P[a][s] for s in seeds], axis=1).mean(axis=1).loc[ix].values; eb = pd.concat([P[b][s] for s in seeds], axis=1).mean(axis=1).loc[ix].values
        B = rng.integers(0, len(y), (2000, len(y)))
        for k in ("acc2", "mae", "corr"):
            ta = [met(y, P[a][s].loc[ix].values)[k] for s in seeds]; tb = [met(y, P[b][s].loc[ix].values)[k] for s in seeds]
            d = np.array([met(y[i], ea[i])[k] - met(y[i], eb[i])[k] for i in B[:500]])
            bp = min(1, max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / 500))
            out.append(f"{sn}.{k} {np.mean(ta) - np.mean(tb):+.4f} (t p={ttest_rel(ta, tb).pvalue:.3f}; boot p={bp:.3f})")
    print(f"\n{a} vs {b} ({len(seeds)} seeds):\n  " + "\n  ".join(out))
